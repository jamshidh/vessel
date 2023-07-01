{-# LANGUAGE RecordWildCards #-}

module Model.Tensor (
  Tensor(..),
  tensorToFloatArray,
  tensorToFloatList
  ) where

import Control.Monad
import Data.Binary
import Data.Binary.Get
import Data.Bits
import Data.ByteString (ByteString)
import qualified Data.ByteString as B
import qualified Data.ByteString.Internal as B
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BC
import Data.Int
import Data.List (intercalate)
import qualified Data.Vector.Storable as V

import Format

data TensorType =
  Q4_0
  | F32 deriving (Show)

instance Binary TensorType where
  get = do
    v <- getInt32le
    case v of
      0 -> return F32
      2 -> return Q4_0
      _ -> error $ "bad tensor type number: " ++ show v
  put = undefined
  

blockSize :: TensorType -> Int
blockSize Q4_0 = 32
blockSize F32 = 1
--blockSize v = error $ "unknown blockSize: " ++ show v

typeSize :: TensorType -> Int
typeSize Q4_0 = 20
typeSize F32 = 4
--typeSize v = error $ "unknown numberElements: " ++ show v

data GenericTensor =
  GenericTensor {
    fType :: TensorType,
    dim_num_elems :: [Int32],
    name :: String,
--    elems :: [Int32le]
    elems :: BC.ByteString
  } deriving (Show)

instance Format Tensor where
  format Tensor{..} =
    "Tensor (" ++ show fType ++ "): "
    ++ "[" ++ intercalate " ⨯ " (map show dim_num_elems) ++ "] "
    ++ name ++ "\n"
    ++ "  elems = " ++ show (B16.encode $ BC.take 100 elems)



instance Binary Tensor where
  get = do
    n_dims' <- getInt32le
    len' <- getInt32le
    ft <- get
    dim_num_elems' <- replicateM (fromIntegral n_dims') getInt32le
    name' <- getByteString (fromIntegral len')
--    elems' <- replicateM (fromIntegral (product dim_num_elems')) get
    elems' <- getByteString (typeSize ft * fromIntegral (product dim_num_elems')`div` blockSize ft)
    return $ Tensor ft dim_num_elems' (BC.unpack name') elems'
  put = undefined
  
tensorToFloatArray :: Tensor -> [[Float]]
tensorToFloatArray t@Tensor{fType=F32} =
  let width = dim_num_elems t !! 0
      height = dim_num_elems t !! 1
  in map (\i -> V.toList $ bytesToFloats $ B.take (fromIntegral $ 4 * height * i) $ B.drop (fromIntegral $ 4 * height * (i+1)) $ elems t) [0..width-1]
tensorToFloatArray t@Tensor{fType=Q4_0} = 
  let width = dim_num_elems t !! 0
  in map (getRow t . fromIntegral) [0..width-1]

  
tensorToFloatList :: Tensor -> [Float]
tensorToFloatList Tensor{..} | length dim_num_elems /= 1 = error "You can't convert a matrix to a vector"
tensorToFloatList t@Tensor{fType=F32} = -- TODO check size matches
  V.toList $ bytesToFloats $ elems t
tensorToFloatList t@Tensor{fType=Q4_0} = getRow t 0


splitInt8IntoNibbles :: Word8 -> [Int8]
splitInt8IntoNibbles v = [(\x -> x-8) $ fromIntegral $ v .&. 0xf, (\x -> x-8) $ fromIntegral $ v `shiftR` 4]


bytesToFloats :: ByteString -> V.Vector Float
bytesToFloats = V.unsafeCast . aux . B.toForeignPtr
  where aux (fp,offset,len) = V.unsafeFromForeignPtr fp offset len

blockToFloats :: ByteString -> [Float]
blockToFloats theBlock = 
  let (dBytes, bytes) = B.splitAt 4 theBlock
      d = bytesToFloats dBytes
      theNibbles = concat $ map splitInt8IntoNibbles $ B.unpack bytes
  in map (*(V.head d::Float)) $ map fromIntegral theNibbles

splitIntoBlocks :: ByteString -> [ByteString]
splitIntoBlocks x | B.length x == 0 = []
splitIntoBlocks x = first:splitIntoBlocks rest
  where (first, rest) = B.splitAt 20 x

bytesForRow :: Tensor -> Int -> ByteString
bytesForRow Tensor{..} i = B.take numberOfRowBytes $ B.drop (i*numberOfRowBytes) elems
  where rowHeight = fromIntegral $ dim_num_elems !! 0
        numberOfRowBytes = rowHeight * 20 `div` 32
  

getRow :: Tensor -> Int -> [Float]
getRow tensor = concat . map blockToFloats . splitIntoBlocks . bytesForRow tensor


data Matrix = Matrix [[Float]] | QuantizedMatrix ByteString

data Vector = Vector [Float] | QuantizedVector ByteString

