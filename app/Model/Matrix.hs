{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}

module Model.Matrix (
  Matrix(..),
  width,
  height,
  unMatrix,
  tensorToMatrix,
  quantizeMatrix
  ) where

import Control.DeepSeq
import Control.Monad
import Data.ByteString (ByteString)
import qualified Data.ByteString as B
import Data.List (intercalate)
import qualified Data.Vector.Storable as V
import GHC.Generics

import Model.Float ()
import Model.GenericTensor
import Model.Vector

import Format


data Matrix = Matrix [Vector] |
              QuantizedMatrix [QuantizedVector] deriving (Generic, NFData)

unMatrix :: Matrix -> [[Float]]
unMatrix (Matrix m) = m
unMatrix (QuantizedMatrix _) = error "unMatrix not defined for QuantizedMatrix"

height :: Matrix -> Int
height (Matrix []) = 0
height (Matrix m) = length $ head m
height (QuantizedMatrix m) = quantizedVectorLength $ head m

width :: Matrix -> Int
width (Matrix m) = length m
width (QuantizedMatrix m) = length m



formatHeight :: Int
formatHeight = 10

formatWidth :: Int
formatWidth = 5



instance Format Matrix where
  format (Matrix []) = "<empty matrix>"
  format (Matrix x) = "[" ++ show (length x) ++ " x " ++ show (length $ head x) ++ "] (sum=" ++ format(sum (join x)) ++ ")\n"
  --             ++ unlines (map (("    " ++) . show . take formatWidth) (take formatWidth x))
                      ++ unlines (map showLine (take formatHeight x))
                      ++ (if length x > formatHeight then "    ....(etc)" else "")
    where showLine v = (++ (if length v > formatWidth then " | ...." else "")) . ("    " ++) . intercalate " | " . map format . take formatWidth $ v
--    where showLine v = (++ (if length v > formatWidth then " | ...." else "")) . ("    " ++) . intercalate " | " . map format $ v
  format m@QuantizedMatrix{} = "QuantizedMatrix [" ++ show (height m) ++ " x " ++ show (width m) ++ "]"

instance Format [Matrix] where
  --format x = "(" ++ format (sum (join $ map join $ transpose $ map unMatrix x)) ++ ") head = " ++ format (head x)
  format x = "(" ++ format (sum (join $ map join $ map unMatrix x)) ++ ") head = " ++ format (head x)



tensorToMatrix :: GenericTensor -> Matrix
tensorToMatrix t@GenericTensor{fType=F32} =
  let width' = dim_num_elems t !! 0
      height' = dim_num_elems t !! 1
  in Matrix $ map (\i -> V.toList $ bytesToFloats $ B.take (fromIntegral $ 4 * height' * i) $ B.drop (fromIntegral $ 4 * height' * (i+1)) $ elems t) [0..width'-1]
tensorToMatrix t@GenericTensor{fType=Q4_0} = 
  let --width' = dim_num_elems t !! 1
      height' = dim_num_elems t !! 0
  in QuantizedMatrix $ map QuantizedVector $ byteStringChunksOf (fromIntegral height' * 20 `div` 32) $ elems t


byteStringChunksOf :: Int -> ByteString -> [ByteString]
byteStringChunksOf _ s | B.length s == 0 = []
byteStringChunksOf i s | B.length s < i = error $ "string length not a multiple of i: remainder = " ++ show (B.length s)
byteStringChunksOf i s = first:byteStringChunksOf i rest
  where (first, rest) = B.splitAt i s

quantizeMatrix :: Matrix -> Matrix
quantizeMatrix (Matrix vectors) = QuantizedMatrix (map quantize vectors)
quantizeMatrix _ = error "unsupported case in quantizeMatrix"







