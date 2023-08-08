
module Model.Int4X32 (
  Int4X32,
  packInt4X32,
  unpackInt4X32,
  byteStringToInt4X32,
  dot_Int4X32
  ) where

import Control.Monad
import Data.Bits
import Data.ByteString (ByteString)
import qualified Data.ByteString as B
import qualified Data.ByteString.Internal as B
import Data.List.Split
import Data.Int
import Data.Word
import Foreign
import System.IO.Unsafe

{-
data Int4X32 = Int4X32 [Int8] deriving (Show)

packInt4X32 :: [Int8] -> Int4X32
packInt4X32 nibbles = Int4X32 nibbles

unpackInt4X32 :: Int4X32 -> [Int8]
unpackInt4X32 (Int4X32 nibbles) = nibbles

byteStringToInt4X32 :: ByteString -> Int4X32
byteStringToInt4X32 bytes = Int4X32 $ concat $ map splitInt8IntoNibbles $ B.unpack bytes

splitInt8IntoNibbles :: Word8 -> [Int8]
splitInt8IntoNibbles v = [(\x -> x-8) $ fromIntegral $ v .&. 0xf, (\x -> x-8) $ fromIntegral $ v `shiftR` 4]

dot_Int4X32 :: Int4X32 -> Int4X32 -> Float

dot_Int4X32 (Int4X32 x) (Int4X32 y) = sum $ zipWith (*) (map fromIntegral x) (map fromIntegral y)
-}





foreign import ccall "dot_Int4X32" c_dot_Int4X32 :: Ptr Word8 -> Ptr Word8 -> Float

data Int4X32 = Int4X32 ByteString deriving (Show)

instance Storable Int4X32 where
  alignment = sizeOf
  sizeOf _ = 16
  poke p (Int4X32 bytes) = do
    let bytes' = B.unpack bytes
        word8_p = castPtr p
    forM_ (zip bytes' [0..]) $ \(b, i) ->
      poke (word8_p `plusPtr` i) b

  peek p = do
    let word8_p = castPtr p
    bytes <- forM [0..15] $ \i -> peek (word8_p `plusPtr` i)
    return $ Int4X32 $ B.pack bytes

packInt4X32 :: [Int8] -> Int4X32
packInt4X32 nibbles = Int4X32 $ B.pack $ map (\[high, low] -> (low `shiftL` 4) + high) $ chunksOf 2 $ map (fromIntegral . (8+)) nibbles

unpackInt4X32 :: Int4X32 -> [Int8]
unpackInt4X32 (Int4X32 nibbles) = map ((\x -> x - 8) . fromIntegral) $ concat $ map (\x -> [x .&. 0xf, x `shiftR` 4]) $ B.unpack nibbles

byteStringToInt4X32 :: ByteString -> Int4X32
byteStringToInt4X32 bytes = Int4X32 bytes

dot_Int4X32 :: Int4X32 -> Int4X32 -> Float
dot_Int4X32 (Int4X32 x) (Int4X32 y) = 
  let (x_fp, _, _) = B.toForeignPtr x
      (y_fp, _, _) = B.toForeignPtr y
  in unsafePerformIO $
  withForeignPtr x_fp $ \x_p ->
  withForeignPtr y_fp $ \y_p ->
    return $ c_dot_Int4X32 x_p y_p
