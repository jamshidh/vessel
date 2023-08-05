
module Model.Int4X32 (
  Int4X32,
  packInt4X32,
  unpackInt4X32,
  byteStringToInt4X32,
  dot_Int4X32
  ) where

import Data.Bits
import Data.ByteString (ByteString)
import qualified Data.ByteString as B
import Data.List.Split
import Data.Int
import Data.Word


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





{-

data Int4X32 = Int4X32 ByteString deriving (Show)

packInt4X32 :: [Int8] -> Int4X32
packInt4X32 nibbles = Int4X32 $ B.pack $ map (\[high, low] -> (high `shiftL` 4) + low) $ chunksOf 2 $ map (fromIntegral . (8+)) nibbles

unpackInt4X32 :: Int4X32 -> [Int8]
unpackInt4X32 (Int4X32 nibbles) = map ((\x -> x - 8) . fromIntegral) $ concat $ map (\x -> [x `shiftR` 4, x .&. 0xf]) $ B.unpack nibbles

byteStringToInt4X32 :: ByteString -> Int4X32
byteStringToInt4X32 bytes = Int4X32 bytes

--splitInt8IntoNibbles :: Word8 -> [Int8]
--splitInt8IntoNibbles v = [(\x -> x-8) $ fromIntegral $ v .&. 0xf, (\x -> x-8) $ fromIntegral $ v `shiftR` 4]

dot_Int4X32 :: Int4X32 -> Int4X32 -> Float

dot_Int4X32 x y = sum $ zipWith (*) (map fromIntegral $ unpackInt4X32 x) (map fromIntegral $ unpackInt4X32 y)

-}
