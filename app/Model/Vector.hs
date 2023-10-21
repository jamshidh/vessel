{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Model.Vector (
  Vector,
  QuantizedBlock(..),
  QuantizedVector(..),
  UnpackedQuantizedVector,
  bytesToFloats,
  tensorToVector,
  quantizedVectorLength,
  quantize,
  dequantize,
  unpackedQuantizedVectorToByteString,
  dot,
  zipFold,
  slowDot,
  quantized_vector_dot,
  fusionMultiplySum
  ) where

import Control.DeepSeq
import Data.ByteString (ByteString)
import qualified Data.ByteString as B
import qualified Data.ByteString.Internal as B
import Data.List.Split
import qualified Data.Vector.Storable as V
import Data.Word
import Foreign
import Foreign.C.String
import GHC.Generics
import System.IO.Unsafe


import Format

import Model.GenericTensor
import Model.Int4X32

type Vector = [Float]

type UnpackedQuantizedVector = V.Vector QuantizedBlock

data QuantizedVector = QuantizedVector ByteString deriving (Show, Generic, NFData)

--instance Format Vector where
--  format x = "[" ++ show (length x) ++ "]\n"
--                      ++ show (take 10 x)

instance Format QuantizedVector where                      
  format v = "QuantizedVector [" ++ show (quantizedVectorLength v) ++ "]"

quantizedVectorLength :: QuantizedVector -> Int
quantizedVectorLength (QuantizedVector x) = 32 * B.length x `div` 20


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

quantizedBlockToFloats :: QuantizedBlock -> [Float]
quantizedBlockToFloats (QuantizedBlock theFloat theNibbles) = 
  map (* theFloat) $ map fromIntegral $ unpackInt4X32 theNibbles

quantize_single_block :: [Float] -> QuantizedBlock
quantize_single_block floats | length floats /= 32 = error $ "quantization blocks must have length 32, actually is " ++ show (length floats)
quantize_single_block floats =
  QuantizedBlock d $ packInt4X32 nibbles
  where d = maximum (map abs floats)/7
        inverse_d = 1/d
        nibbles = map ((\v -> v-8) . truncate . (8.5+) . (inverse_d *)) floats

quantize :: Vector -> QuantizedVector
--quantize x | trace ("quantizing: " ++ show (length x)) False = undefined
quantize floats = QuantizedVector $ unpackedQuantizedVectorToByteString $ V.fromList $ map quantize_single_block $ chunksOf 32 floats

dequantize :: QuantizedVector -> Vector
dequantize (QuantizedVector bytes) = concat . map quantizedBlockToFloats . V.toList $ byteStringToUnpackedQuantizedBlocks bytes



unpackedQuantizedVectorToByteString :: UnpackedQuantizedVector -> ByteString
unpackedQuantizedVectorToByteString v =
  let (p, l) = V.unsafeToForeignPtr0 v
  in
    unsafePerformIO $
    withForeignPtr p $ \p' -> 
                         B.packCStringLen (castPtr p', l * sizeOf (undefined :: QuantizedBlock))

byteStringToUnpackedQuantizedBlocks :: ByteString -> UnpackedQuantizedVector
byteStringToUnpackedQuantizedBlocks theData = V.fromList $ map parseQuantizedBlock $ splitIntoBlocks theData
  where
    parseQuantizedBlock :: ByteString -> QuantizedBlock
    parseQuantizedBlock theBlock = 
      let (dBytes, bytes) = B.splitAt 4 theBlock
          d = bytesToFloats dBytes
          theNibbles = byteStringToInt4X32 bytes
      in QuantizedBlock (V.head d) theNibbles
    splitIntoBlocks :: ByteString -> [ByteString]
    splitIntoBlocks x | B.length x == 0 = []
    splitIntoBlocks x = first:splitIntoBlocks rest
      where (first, rest) = B.splitAt 20 x

zipFold :: (a -> b -> c -> c) -> c -> [a] -> [b] -> c
zipFold _ initVal [] [] = initVal
zipFold f initVal (firstx:restx) (firsty:resty) =
  let newVal = f firstx firsty initVal
  in zipFold f newVal restx resty
zipFold _ _ _ _ = error "mismatched array sizes in call to zipFold"

foreign import ccall "fusionMultiplySum" fusionMultiplySum :: Float -> Float -> Double -> Double
foreign import ccall "fusionMultiplySumAllFloat" fusionMultiplySumAllFloat :: Float -> Float -> Float -> Float

foreign import ccall "vector_dot" vector_dot :: Int -> CString -> CString -> Float


slowDot :: [Float] -> [Float] -> Float
--slowDot x y = realToFrac $ sum $ zipWith (*) (map realToFrac x) (map realToFrac y :: [Double])
--slowDot x y = sum $ zipWith (*) x y

slowDot x y = zipFold (\v1 v2 s -> fusionMultiplySumAllFloat v1 v2 s) (0.0::Float) x y

dot :: Vector -> Vector -> Float
--dot x y | trace ("dot, length x = " ++ show (length x) ++ ", length y = " ++ show (length y)) False = undefined
dot x y | length x /= length y = error $ "dot product lengths do not match: " ++ show (length x) ++ "/=" ++ show (length y)
dot x y = realToFrac $ zipFold fusionMultiplySum (0.0::Double) x y
  --sum $ zipWith (*) (map realToFrac x::[Double]) (map realToFrac y::[Double])

quantized_vector_dot :: QuantizedVector -> QuantizedVector -> Float
--quantized_block_dot (QuantizedBlock f1 n1) (QuantizedBlock f2 n2) | trace ("f1 = " ++ format f1 ++ ", f2 = " ++ format f2 ++ ", n1 = " ++ show n1 ++ ", n2 = " ++ show n2 ++ ", int dot = " ++ show (sum $ zipWith (*) n1 n2)) False = undefined
quantized_vector_dot v1 v2 | quantizedVectorLength v1 /= quantizedVectorLength v2 = error "vector lengths different in call to quantized_vector_dot"
quantized_vector_dot (QuantizedVector v1) (QuantizedVector v2) =
  unsafePerformIO $
  B.useAsCStringLen v1 $ \(p1, _) ->
  B.useAsCStringLen v2 $ \(p2, _) ->
                      return $ vector_dot (B.length v1 `div` 20) p1 p2


