{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}

module Model.Matrix (
  Matrix(..),
  matrixVectors,
  matrixMap,
  width,
  height,
  tensorToMatrix,
  getRow,
  matMul
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

matrixVectors :: Matrix -> [Vector]
matrixVectors (Matrix vectors) = vectors
matrixVectors _ = error "matrixVectors not defined for QuantizedMatrix"

qMatrixVectors :: Matrix -> [QuantizedVector]
qMatrixVectors (QuantizedMatrix matrixData) = matrixData
qMatrixVectors _ = error "trying to convert non quantized Matrix to quantized vectors"

matrixMap :: (Float -> Float) -> Matrix -> Matrix
matrixMap f (Matrix m) = Matrix $ map (map f) m
matrixMap _ (QuantizedMatrix _) = error "matrixMap not defined for QuantizedMatrix"

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
  format x = "(" ++ format (sum (join $ map join $ map matrixVectors x)) ++ ") head = " ++ format (head x)



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

getRow :: Matrix -> Int -> Vector
--getRow m i | trace ("getRow: " ++ format m ++ " " ++ show i) False = undefined
getRow (QuantizedMatrix matrixData) i = dequantize $ matrixData !!  i 
getRow _ _ = error "getRow not definted for non-quantized Matrix"

matMul :: Matrix -> Matrix -> Matrix
--matMul x y | trace ("multiplying: " ++ formatShortMatrix x ++ " * " ++ formatShortMatrix y ++ ", num ops = " ++ show (height x * width x * width y) ++ ", num vec ops = " ++ show (width x * width y)) False = undefined
matMul x y | height x /= height y = error $ "matrix heights don't match:\n" ++
             "x size is: [" ++ show (height x) ++ " x " ++ show (width x) ++ "]\n" ++
             "y size is: [" ++ show (height y) ++ " x " ++ show (width y) ++ "]\n" ++
             show (height x) ++ " /= " ++ show (height y)
matMul x@(Matrix xVals) (Matrix yVals) | height x < width x =
  Matrix $ map (\yRow -> map (\xCol -> xCol `slowDot` yRow) xVals) yVals
matMul x@(Matrix _) y@(Matrix _) =
  Matrix $ map (\yRow -> map (\xCol -> xCol `dot` yRow) $ matrixVectors x) $ matrixVectors y
matMul x@(QuantizedMatrix _) y@(Matrix _) =
  Matrix $ map (\yRow -> map (\xCol -> xCol `quantized_vector_dot` yRow) $ qMatrixVectors x) $ qMatrixVectors $ quantizeMatrix y
matMul _ _ = error "unsupported case called in matMul"
