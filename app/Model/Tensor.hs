{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Model.Tensor (
  GenericTensor(..),
  QuantizedBlock(..),
  Matrix(..),
  Vector(..),
  width,
  height,
  tensorToMatrix,
  tensorToVector,
  getRow,
  bytesToFloats,
  splitIntoBlocks,
  splitIntoQuantizedBlocks
  ) where

import Control.DeepSeq
import Control.Monad
import Data.Binary
import Data.Binary.Get
import Data.ByteString (ByteString)
import qualified Data.ByteString as B
import qualified Data.ByteString.Internal as B
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BC
import Data.Int
import Data.List (intercalate)
import qualified Data.Vector.Storable as V
import Foreign.Ptr
import Foreign.Storable
import GHC.Generics

import Format
import Model.Float ()
import Model.Int4X32

--import Debug.Trace

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

instance Format GenericTensor where
  format GenericTensor{..} =
    "Tensor (" ++ show fType ++ "): "
    ++ "[" ++ intercalate " ⨯ " (map show dim_num_elems) ++ "] "
    ++ name ++ "\n"
    ++ "  elems = " ++ show (B16.encode $ BC.take 100 elems)



instance Binary GenericTensor where
  get = do
    n_dims' <- getInt32le
    len' <- getInt32le
    ft <- get
    dim_num_elems' <- replicateM (fromIntegral n_dims') getInt32le
    name' <- getByteString (fromIntegral len')
--    elems' <- replicateM (fromIntegral (product dim_num_elems')) get
    elems' <- getByteString (typeSize ft * fromIntegral (product dim_num_elems')`div` blockSize ft)
    return $ GenericTensor ft dim_num_elems' (BC.unpack name') elems'
  put = undefined

tensorToMatrix :: GenericTensor -> Matrix
tensorToMatrix t@GenericTensor{fType=F32} =
  let width' = dim_num_elems t !! 0
      height' = dim_num_elems t !! 1
  in Matrix $ map (\i -> V.toList $ bytesToFloats $ B.take (fromIntegral $ 4 * height' * i) $ B.drop (fromIntegral $ 4 * height' * (i+1)) $ elems t) [0..width'-1]
tensorToMatrix t@GenericTensor{fType=Q4_0} = 
  let --width' = dim_num_elems t !! 1
      height' = dim_num_elems t !! 0
  in QuantizedMatrix (map splitIntoQuantizedBlocks $ byteStringChunksOf (fromIntegral height' * 20 `div` 32) $ elems t)

byteStringChunksOf :: Int -> ByteString -> [ByteString]
byteStringChunksOf _ s | B.length s == 0 = []
byteStringChunksOf i s | B.length s < i = error $ "string length not a multiple of i: remainder = " ++ show (B.length s)
byteStringChunksOf i s = first:byteStringChunksOf i rest
  where (first, rest) = B.splitAt i s



tensorToVector :: GenericTensor -> Vector
tensorToVector GenericTensor{..} | length dim_num_elems /= 1 = error "You can't convert a matrix to a vector"
tensorToVector t@GenericTensor{fType=F32} = -- TODO check size matches
  Vector $ V.toList $ bytesToFloats $ elems t
tensorToVector t@GenericTensor{fType=Q4_0} = QuantizedVector $ splitIntoQuantizedBlocks $ elems t -- $ getRow t 0

--tensorToVector = Vector . tensorToFloatList 



bytesToFloats :: ByteString -> V.Vector Float
bytesToFloats = V.unsafeCast . aux . B.toForeignPtr
  where aux (fp,offset,len) = V.unsafeFromForeignPtr fp offset len
{-
blockToFloats :: ByteString -> [Float]
blockToFloats theBlock = 
  let (dBytes, bytes) = B.splitAt 4 theBlock
      d = bytesToFloats dBytes
      theNibbles = concat $ map splitInt8IntoNibbles $ B.unpack bytes
  in map (*(V.head d::Float)) $ map fromIntegral theNibbles
-}

quantizedBlockToFloats :: QuantizedBlock -> [Float]
quantizedBlockToFloats (QuantizedBlock theFloat theNibbles) = 
  map (* theFloat) $ map fromIntegral $ unpackInt4X32 theNibbles


splitIntoBlocks :: ByteString -> [ByteString]
splitIntoBlocks x | B.length x == 0 = []
splitIntoBlocks x = first:splitIntoBlocks rest
  where (first, rest) = B.splitAt 20 x
{-
bytesForRow :: Matrix -> Int -> ByteString
bytesForRow QuantizedMatrix{..} i = B.take numberOfRowBytes $ B.drop (i*numberOfRowBytes) matrixData
  where numberOfRowBytes = matrixHeight * 20 `div` 32
bytesForRow Matrix{} _ = error "bytesForRow only implemented for Quantized matrix"
-}


getRow :: Matrix -> Int -> Vector
--getRow m i | trace ("getRow: " ++ format m ++ " " ++ show i) False = undefined
getRow (QuantizedMatrix matrixData) i = Vector . concat . map quantizedBlockToFloats . V.toList . (matrixData !!) $ i -- concat . map blockToFloats . splitIntoBlocks . bytesForRow m
getRow _ _ = error "getRow not definted for non-quantized Matrix"

data QuantizedBlock = QuantizedBlock Float Int4X32 deriving (Show)

instance Storable QuantizedBlock where
  sizeOf _ = 20
  alignment = sizeOf
  peek p = do
    f <- peek (castPtr p)
    nibbles <- peek (castPtr $ (castPtr p::Ptr Word8) `plusPtr` 4)
    return $ QuantizedBlock f nibbles
    
  poke p (QuantizedBlock f nibbles) = do
    poke (castPtr p) f
    poke (castPtr $ (castPtr p::Ptr Word8) `plusPtr` 4) nibbles

splitIntoQuantizedBlocks :: ByteString -> (V.Vector QuantizedBlock)
splitIntoQuantizedBlocks theData = V.fromList $ map parseQuantizedBlock $ splitIntoBlocks theData
  where
    parseQuantizedBlock :: ByteString -> QuantizedBlock
    parseQuantizedBlock theBlock = 
      let (dBytes, bytes) = B.splitAt 4 theBlock
          d = bytesToFloats dBytes
          theNibbles = byteStringToInt4X32 bytes
      in QuantizedBlock (V.head d) theNibbles




data Matrix = Matrix [[Float]] |
              QuantizedMatrix [V.Vector QuantizedBlock] deriving (Generic, NFData)

height :: Matrix -> Int
height (Matrix []) = 0
height (Matrix m) = length $ head m
height (QuantizedMatrix m) = 32 * V.length (head m)

width :: Matrix -> Int
width (Matrix m) = length m
width (QuantizedMatrix m) = length m




data Vector = Vector [Float] | QuantizedVector (V.Vector QuantizedBlock) deriving (Show, Generic, NFData)

instance Format Vector where
  format (Vector x) = "[" ++ show (length x) ++ "]\n"
                      ++ show (take 10 x)
  format (QuantizedVector theData) = "QuantizedVector [" ++ show (V.length theData * 32) ++ "]"

instance Format Matrix where
  format (Matrix []) = "<empty matrix>"
  format (Matrix x) = "[" ++ show (length x) ++ " x " ++ show (length $ head x) ++ "] (sum=" ++ format(sum (join x)) ++ ")\n"
  --             ++ unlines (map (("    " ++) . show . take 5) (take 5 x))
                      ++ unlines (map showLine (take 5 x))
                      ++ (if length x > 5 then "    ....(etc)" else "")
    where showLine v = (++ (if length v > 5 then " | ...." else "")) . ("    " ++) . intercalate " | " . map format . take 5 $ v
--    where showLine v = (++ (if length v > 5 then " | ...." else "")) . ("    " ++) . intercalate " | " . map format $ v
  format m@QuantizedMatrix{} = "QuantizedMatrix [" ++ show (height m) ++ " x " ++ show (width m) ++ "]"

