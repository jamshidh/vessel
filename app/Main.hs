{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE HexFloatLiterals #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

--import Control.DeepSeq
import Control.Monad
import Data.Binary
import qualified Data.ByteString.Lazy as BL
import Data.IORef
import Data.List (transpose)
import Data.List.Split
import qualified Data.Vector.Storable as V
import Foreign.Ptr
import Numeric.Half
import System.IO.Unsafe

import Format

import Model.Float ()
import Model.Int4X32
import Model.Model
import Model.Tensor
import Model.Token
import Rope

import Debug.Trace

numberOfHeads :: Int
numberOfHeads = 32

traceItem :: Format a => String -> a -> a
traceItem name item = trace (name ++ ": " ++ format item) item


main :: IO ()
main = do
  --let size = 48000000
  --putStrLn $ format $ QuantizedMatrix [quantize $ take size [1..]] size 1 `matMul` Matrix [take size [1..]]
  doit

doit :: IO ()
doit = do
  x <- BL.readFile "ggml-alpaca-7b-q4.bin"
  let rawModel = decode x
  let model :: Model
      model = rawModelToModel rawModel

  --putStrLn $ model `deepseq` "abcd"
  

  let phrase = " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"

  let theTrie = tokensToTokenTrie $ tokens model

  let phraseTokens = 1:tokenize theTrie phrase
{-  
  let phraseTokensFromCPP = [1, 13866, 338, 385, 15278, 393, 16612, 263, 3414, 29889, 14350, 263, 2933, 393, 8210, 368, 4866, 29879, 278, 2009, 29889, 13, 13]

  putStrLn $ "phraseTokens: " ++ show phraseTokens
  putStrLn $ "phraseTokensFromCPP: " ++ show phraseTokensFromCPP
  
  putStrLn $ show $ map (\i -> (i, format $ tokens model !! i)) phraseTokens
  putStrLn $ show $ map (\i -> (i, format $ tokens model !! i)) phraseTokensFromCPP
-}

  putStrLn $ "phrase tokens: " ++ show phraseTokens
  
--  putStrLn $ format rawModel

  let embd = [0,1,2,3]
--  let embd = take 9 $ drop 9 phraseTokens

  let inputLayer = vectorsToMatrix $ map (getRow $ tokenEmbeddings model) embd

  value <- newIORef inputLayer
  
  putStrLn $ "inputLayer = " ++ format inputLayer

--  let outputLayer = applyPipeline (map processLayer $ reverse $ layers model) inputLayer

  writeIORef value inputLayer


  forM_ (layers model) $ \layer -> do
    currentValue <- readIORef value
    let output = processLayer layer currentValue
    writeIORef value output
    --putStrLn $ "output = " ++ format output


  outputLayer <- readIORef value

  putStrLn $ "outputLayer = " ++ format outputLayer

  let normalizedOutputLayer = meanNorm outputLayer

  putStrLn $ "normalizedOutputLayer = " ++ format normalizedOutputLayer

  let normalizedOutputLayer2 = replicateVector (norm model) (width normalizedOutputLayer) `simpleElementMul` normalizedOutputLayer

  putStrLn $ "normalizedOutputLayer2 = " ++ format normalizedOutputLayer2

  let output' = output model `matMul` normalizedOutputLayer2

  putStrLn $ "output' = " ++ format output'



vectorsToMatrix :: [Vector] -> Matrix
vectorsToMatrix vectors = 
  case ([ v | Vector v <- vectors ], [ v | QuantizedVector v <- vectors ]) of
    (vectors', []) -> Matrix vectors'
    ([], quantizedVectors) -> QuantizedMatrix quantizedVectors
    _ -> error "error calling vectorsToMatrix: list of Vectors contains mixed quantized and non-quantized values."

matrixVectors :: Matrix -> [Vector]
matrixVectors (Matrix vectors) = map Vector vectors
matrixVectors (QuantizedMatrix matrixData) = map QuantizedVector matrixData

{-
applyPipeline :: [(a->a)] -> a -> a
applyPipeline = flip $ foldr ($)
-}

processLayer :: Layer -> Matrix -> Matrix
processLayer layer inputLayer =
  let normalized = meanNorm (traceItem ("[" ++ show (layerNumber layer) ++ "] inputLayer") inputLayer)
      --normalized2 = map (zipWith (*) $ attention_norm layer) normalized
      normalized2 = replicateVector (attention_norm layer) (width normalized) `simpleElementMul` normalized
      afterAttention = selfAttention layer normalized2
      inputFF = afterAttention `matAdd` inputLayer -- normalized2
      outputFF = feedForward layer inputFF
      outputLayer = outputFF `matAdd` inputFF
  in outputLayer
  
instance Format [Matrix] where
  --format x = "(" ++ format (sum (join $ map join $ transpose $ map unMatrix x)) ++ ") head = " ++ format (head x)
  format x = "(" ++ format (sum (join $ map join $ map unMatrix x)) ++ ") head = " ++ format (head x)

selfAttention :: Layer -> Matrix -> Matrix
selfAttention Layer{..} inputSA =
  let
      qCur = attention_wq `matMul` inputSA  -- [4x4096] = [??] * [??]
      kCur = attention_wk `matMul` inputSA
      vCur = attention_wv `matMul` inputSA
      headSize = height kCur `div` numberOfHeads
      kBlobs :: [Matrix]
      kBlobs = map (Matrix . embedPositions) $ transpose $ map (chunksOf headSize) $ unMatrix kCur
      qBlobs :: [Matrix]
      qBlobs = map (Matrix . embedPositions) $ transpose $ map (chunksOf headSize) $ unMatrix qCur
      vBlobs :: [Matrix]
      vBlobs = map Matrix $ transpose $ map (chunksOf headSize) $ unMatrix vCur
      kqs :: [Matrix]
      kqs = zipWith matMul kBlobs qBlobs -- 32 times: [4 x 4] = [4 x 128] * [4 x 128]
      kqs_scaled :: [Matrix]
      kqs_scaled = --map (matrixMap (/sqrt 128.0)) kqs
                   map (matrixMap (* (0x1.6a09e6p-4))) kqs
      kqs_masked :: [Matrix]
      kqs_masked = map filterUpperDiagonal kqs_scaled
      kqs_softmax :: [Matrix]
      kqs_softmax = map (buildMatrixFromRows . map softMax . matrixRows) kqs_masked
      kqv :: [Matrix]
      kqv = zipWith matMul (map transposeMatrix vBlobs) kqs_softmax -- 32 times: [128 x 4] = [4 x 128] * [4 x 4]
      kqv_smashed = transposeMatrix (matrixConcat (map transposeMatrix kqv)) -- [??] = [4 x 4096] * [4096 x 4]
      output = attention_wo `matMul` kqv_smashed
  in output

feedForward :: Layer -> Matrix -> Matrix
feedForward Layer{..} inpFF = 
  let cur1 = meanNorm inpFF
      cur2 = replicateVector ffn_norm (width cur1) `simpleElementMul` cur1           --map (zipWith (*) (ffn_norm layer)) cur1
      tmp = feed_forward_w3 `matMul` cur2
      cur3 = feed_forward_w1 `matMul` cur2
--      cur4 = matrixMap silu cur3
      cur4 = matrixMap (fromHalf . toHalf . silu . fromHalf . toHalf) cur3
      cur5 = cur4 `simpleElementMul` tmp
      cur6 = feed_forward_w2 `matMul` cur5
  in cur6

unMatrix :: Matrix -> [[Float]]
unMatrix (Matrix m) = m
unMatrix (QuantizedMatrix _) = error "unMatrix not defined for QuantizedMatrix"

matrixRows :: Matrix -> [Vector]
matrixRows (Matrix rows) = map Vector rows
matrixRows (QuantizedMatrix _) = error "matrixRows not defined for QuantizedMatrix"


buildMatrixFromRows :: [Vector] -> Matrix
buildMatrixFromRows rows =
  Matrix $ [row | Vector row <- rows]

matrixConcat :: [Matrix] -> Matrix
matrixConcat matrixList = Matrix $ concat $ map unMatrix matrixList


silu :: Float -> Float
silu x =
  let x_double :: Double
      x_double = realToFrac x
  in realToFrac $ x_double/(1+exp (-x_double))

replicateVector :: Vector -> Int -> Matrix
replicateVector (Vector vals) i = Matrix $ replicate i vals
replicateVector (QuantizedVector _) _ = error "replicateVector not defined for QuantizedVector"

transposeMatrix :: Matrix -> Matrix
transposeMatrix (Matrix m) = Matrix $ transpose m
transposeMatrix (QuantizedMatrix _) = error "transposeMatrix not defined for QuantizedMatrix"

simpleElementMul :: Matrix -> Matrix -> Matrix
simpleElementMul x y | (height x /= height y) || (width x /= width y) = error "mismsatched matrix sizes"
simpleElementMul (Matrix x) (Matrix y) = Matrix $ zipWith (zipWith (*)) x y
simpleElementMul _ _ = error "simpleElementMul not defined for QuantizedMatrix"


matrixMap :: (Float -> Float) -> Matrix -> Matrix
matrixMap f (Matrix m) = Matrix $ map (map f) m
matrixMap _ (QuantizedMatrix _) = error "matrixMap not defined for QuantizedMatrix"

exp' :: Float -> Float
exp' = fromHalf . toHalf . exp . fromHalf . toHalf

softMax :: Vector -> Vector
softMax (Vector theRow) = Vector . normalize . map (exp' . (\v -> v - maximum theRow)) $ theRow
softMax (QuantizedVector _) = error "softMax not defined for QuantizedVector"

filterUpperDiagonal :: Matrix -> Matrix
filterUpperDiagonal (Matrix theMatrix) = Matrix $ map (\(theRow, i) -> filterAfter i theRow) $ zip theMatrix [1..]
filterUpperDiagonal (QuantizedMatrix _) = error "filterUpperDiagonal not defined for QuantizedMatrix"


filterAfter :: Int -> [Float] -> [Float]
filterAfter i theRow = take i theRow ++ replicate (length theRow - i) (-inf)
  where inf = 1/0

normalize :: [Float] -> [Float]
normalize values =
  let theSum = sum $ map realToFrac values :: Double
      scaleVal = realToFrac $ 1/theSum :: Float
  in map (* scaleVal) values


matAdd :: Matrix -> Matrix -> Matrix
matAdd (Matrix x) (Matrix y) = Matrix $ zipWith vectAdd x y
matAdd _ _ = error "matAdd not defined for QuantizedMatrix"

vectAdd :: [Float] -> [Float] -> [Float]
vectAdd = zipWith (+)
{-
formatShortMatrix :: Matrix -> String
formatShortMatrix m =
  (
    case m of
      (Matrix _) -> ""
      (QuantizedMatrix _ _ _) -> "Q"
  )
  ++ "[" ++ show (height m) ++ "x" ++ show (width m) ++ "]"
-}

matMul :: Matrix -> Matrix -> Matrix
--matMul x y | trace ("multiplying: " ++ formatShortMatrix x ++ " * " ++ formatShortMatrix y ++ ", num ops = " ++ show (height x * width x * width y) ++ ", num vec ops = " ++ show (width x * width y)) False = undefined
matMul x y | height x /= height y = error $ "matrix heights don't match: " ++ show (height x) ++ " /= " ++ show (height y)
matMul x@(Matrix xVals) (Matrix yVals) | height x < width x =
  Matrix $ map (\yRow -> map (\xCol -> xCol `slowDot` yRow) xVals) yVals
matMul x@(Matrix _) y@(Matrix _) =
  Matrix $ map (\yRow -> map (\xCol -> xCol `dot` yRow) $ matrixVectors x) $ matrixVectors y
matMul x@(QuantizedMatrix _) y@(Matrix _) =
  Matrix $ map (\yRow -> map (\xCol -> xCol `dot` yRow) $ matrixVectors x) $ matrixVectors $ quantizeMatrix y
matMul _ _ = error "unsupported case called in matMul"

quantizeMatrix :: Matrix -> Matrix
quantizeMatrix (Matrix vectors) = QuantizedMatrix (map quantize vectors)
quantizeMatrix _ = error "unsupported case in quantizeMatrix"

slowDot :: [Float] -> [Float] -> Float
--slowDot x y = realToFrac $ sum $ zipWith (*) (map realToFrac x) (map realToFrac y :: [Double])
--slowDot x y = sum $ zipWith (*) x y

slowDot x y = zipFold (\v1 v2 s -> fusionMultiplySumAllFloat v1 v2 s) (0.0::Float) x y

dot :: Vector -> Vector -> Float
--dot x y | trace ("dot, length x = " ++ show (vectorLength x) ++ ", length y = " ++ show (vectorLength y)) False = undefined
dot x y | vectorLength x /= vectorLength y = error $ "dot product lengths do not match: " ++ show (vectorLength x) ++ "/=" ++ show (vectorLength y)
dot (Vector x) (Vector y) = realToFrac $ zipFold fusionMultiplySum (0.0::Double) x y
  --sum $ zipWith (*) (map realToFrac x::[Double]) (map realToFrac y::[Double])
dot (QuantizedVector x) (Vector y) = x `quantized_vector_dot` quantize y
dot (QuantizedVector x) (QuantizedVector y) = x `quantized_vector_dot` y
dot (Vector x) (QuantizedVector y) = quantize x `quantized_vector_dot` y

zipFold :: (a -> b -> c -> c) -> c -> [a] -> [b] -> c
zipFold _ initVal [] [] = initVal
zipFold f initVal (firstx:restx) (firsty:resty) =
  let newVal = f firstx firsty initVal
  in zipFold f newVal restx resty
zipFold _ _ _ _ = error "mismatched array sizes in call to zipFold"

foreign import ccall "fusionMultiplySum" fusionMultiplySum :: Float -> Float -> Double -> Double
foreign import ccall "fusionMultiplySumAllFloat" fusionMultiplySumAllFloat :: Float -> Float -> Float -> Float

vectorLength :: Vector -> Int
vectorLength (Vector elems) = length elems
vectorLength (QuantizedVector elems) = V.length elems * 32

foreign import ccall "vector_dot" vector_dot :: Int -> Ptr QuantizedBlock -> Ptr QuantizedBlock -> Float

quantized_vector_dot :: V.Vector QuantizedBlock -> V.Vector QuantizedBlock -> Float
--quantized_block_dot (QuantizedBlock f1 n1) (QuantizedBlock f2 n2) | trace ("f1 = " ++ format f1 ++ ", f2 = " ++ format f2 ++ ", n1 = " ++ show n1 ++ ", n2 = " ++ show n2 ++ ", int dot = " ++ show (sum $ zipWith (*) n1 n2)) False = undefined
quantized_vector_dot v1 v2 | V.length v1 /= V.length v2 = error "vector lengths different in call to quantized_vector_dot"
quantized_vector_dot v1 v2 = unsafePerformIO $
  V.unsafeWith v1 $ \p1 ->
  V.unsafeWith v2 $ \p2 ->
                      return $ vector_dot (V.length v1) p1 p2



{-
--quantized_block_dot (QuantizedBlock f1 ints1) (QuantizedBlock f2 ints2) = --traceItem "quantized_block_dot" $ 
--  f1 * f2 * (fromIntegral $ sum $ zipWith (*) ints1 ints2)
  let f1 = V.unsafeCast $ castPtr p1 :: Float
      f2 = V.unsafeCast $ castPtr p2 :: Float
  in f1 * f2 * ints1 `dot_Int4X32` ints2
-}

quantize :: [Float] -> V.Vector QuantizedBlock
--quantize x | trace ("quantizing: " ++ show (length x)) False = undefined
quantize floats = V.fromList $ map quantize_single_block $ chunksOf 32 floats

quantize_single_block :: [Float] -> QuantizedBlock
quantize_single_block floats | length floats /= 32 = error $ "quantization blocks must have length 32, actually is " ++ show (length floats)
quantize_single_block floats =
  QuantizedBlock d $ packInt4X32 nibbles
  where d = maximum (map abs floats)/7
        inverse_d = 1/d
        nibbles = map ((\v -> v-8) . truncate . (8.5+) . (inverse_d *)) floats

meanNorm :: Matrix -> Matrix
meanNorm m@(Matrix theFloats) = 
  let squaresSummed = zipWith ((\v1 v2 -> zipFold fusionMultiplySum (0.0::Double) v1 v2)) theFloats theFloats
      mean = map (/realToFrac (height m)) squaresSummed :: [Double]
      scaleFactors = map realToFrac $ map (1/) $ (map (sqrt . (+1e-5)) mean) :: [Float]
  in Matrix $ zipWith (\sf f -> map (sf*) f) scaleFactors theFloats
meanNorm (QuantizedMatrix _) = error "meanNorm not defined for QuantizedMatrix"
