{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Model.Vector (
  Vector,
  QuantizedBlock(..),
  QuantizedVector(..),
  bytesToFloats,
  tensorToVector,
  quantizedVectorLength
  ) where

import Control.DeepSeq
import Data.ByteString (ByteString)
import qualified Data.ByteString.Internal as B
import qualified Data.Vector.Storable as V
import Data.Word
import Foreign.Ptr
import Foreign.Storable
import GHC.Generics


import Format

import Model.GenericTensor
import Model.Int4X32

type Vector = [Float]

data QuantizedVector = QuantizedVector (V.Vector QuantizedBlock) deriving (Show, Generic, NFData)

--instance Format Vector where
--  format x = "[" ++ show (length x) ++ "]\n"
--                      ++ show (take 10 x)

instance Format QuantizedVector where                      
  format (QuantizedVector theData) = "QuantizedVector [" ++ show (V.length theData * 32) ++ "]"

quantizedVectorLength :: QuantizedVector -> Int
quantizedVectorLength (QuantizedVector x) = 32 * V.length x


data QuantizedBlock = QuantizedBlock Float Int4X32 deriving (Show)

instance Storable QuantizedBlock where
  sizeOf _ = 20
  alignment _ = 1
  peek p = do
    f <- peek (castPtr p)
    nibbles <- peek (castPtr $ (castPtr p::Ptr Word8) `plusPtr` 4)
    return $ QuantizedBlock f nibbles
    
  poke p (QuantizedBlock f nibbles) = do
    poke (castPtr p) f
    poke (castPtr $ (castPtr p::Ptr Word8) `plusPtr` 4) nibbles

tensorToVector :: GenericTensor -> Vector
tensorToVector GenericTensor{..} | length dim_num_elems /= 1 = error "You can't convert a matrix to a vector"
tensorToVector t@GenericTensor{fType=F32} = -- TODO check size matches
  V.toList $ bytesToFloats $ elems t
tensorToVector _ = error "you can't convert a quantized tensor to a non quantized vector"

bytesToFloats :: ByteString -> V.Vector Float
bytesToFloats = V.unsafeCast . aux . B.toForeignPtr
  where aux (fp,offset,len) = V.unsafeFromForeignPtr fp offset len
