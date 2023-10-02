{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE HexFloatLiterals #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

--import Control.DeepSeq
import Control.Monad
import Control.Monad.IO.Class
import qualified Control.Monad.Trans.State as State
import Data.Binary
import qualified Data.ByteString.Lazy as BL
import Data.IORef
import Data.List (sortOn, transpose)
import Data.List.Split
import qualified Data.Vector.Storable as V
import Foreign.Ptr
import Numeric.Half
import System.IO.Unsafe

import Converse

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
  

  let theTrie = tokensToTokenTrie $ tokens model

--  putStrLn $ format rawModel

--  let embd = [0,1,2,3]

  let phraseChunks =
        chunksOf 9 (1:tokenize theTrie " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n")
        ++ chunksOf 9 (1:tokenize theTrie "### Instruction:\n\n1+1\n### Response:\n\n")
        ++ chunksOf 9 (tokenize theTrie "2")

  runConverse $ do
    results <- 
      forM phraseChunks $ \phraseChunk -> do
      tokensWithProbs <- handleNewTokens model $ phraseChunk
      liftIO $ putStrLn $ "most probable next token: " ++ (show . format) (tokens model !! getNextToken tokensWithProbs)
      return tokensWithProbs

    liftIO $ printTopTokens model $ last results

  putStrLn "done"


getNextToken :: [(Int, Float)] -> Int
getNextToken = fst . head --just take the most probable for now

handleNewTokens :: Model -> [Int] -> Converse [(Int, Float)]
handleNewTokens model phrase = do
  (tokenHistory, historyKVs) <- State.get
  let position = length tokenHistory - 1
  liftIO $ putStrLn $ "input phrase = " ++ show (map (format . (tokens model !!)) phrase)
  (Matrix output, extras) <- runNN model position historyKVs phrase

  --putStrLn $ "output logits: " ++ format (last output)

  let tokensWithProbs = logitsToTopProbabilities (tokenHistory ++ phrase) $ zip [0..] $ last output

  State.put (tokenHistory ++ phrase, extras)
  
  return tokensWithProbs


scale :: [Int] -> (Int, Float) -> Float
scale tokenHistoryLast (t, l) =
  let scaleFactor =
        case (t `elem` tokenHistoryLast, l<0.0) of
          (False, _) -> 10
          (_, True) -> 10*1.3
          (_, False) -> 10/1.3
  in l*scaleFactor

logitsToTopProbabilities :: [Int] -> [(Int, Float)] -> [(Int, Float)]
logitsToTopProbabilities tokenHistory tokens'= 
  let tokensScaled = map (\(t, l) -> (t, scale tokenHistoryLast (t, l))) tokens'
      sortedTokens = reverse $ sortOn snd tokensScaled
      topTokensScaled = take 40 sortedTokens
      tokenHistoryLast = --traceItem "tokenHistory"
                         tokenHistory
      allLogits = map snd topTokensScaled
      maxLogit = maximum allLogits
      nonNormalizedProbs = map (fmap (exp . (\v -> v - maxLogit))) topTokensScaled
      nonNormalizedSum = sum $ map snd nonNormalizedProbs
      probs = map (fmap (/nonNormalizedSum)) nonNormalizedProbs
  in probs

printTopTokens :: Model -> [(Int, Float)] -> IO ()
printTopTokens model tokensWithProbs =
  forM_ tokensWithProbs $ \(tokenInt, prob) ->
    putStrLn $ format prob ++ ": (" ++ show tokenInt ++ ") " ++ show (format $ tokens model !! tokenInt)

--instance Format [Int] where
--  format = show

runNN :: Model -> Int -> [HistoryKVs] -> [Int] -> Converse (Matrix, [HistoryKVs])
runNN model startingPosition extras embd = do
  let inputLayer = vectorsToMatrix $ map (getRow $ tokenEmbeddings model) embd

  value <- liftIO $ newIORef inputLayer

--  let outputLayer = applyPipeline (map processLayer $ reverse $ layers model) inputLayer

  allExtras <- 
    forM (zip (layers model) extras) $ \(layer, layerExtra) -> do
      currentValue <- liftIO $ readIORef value
      let (output, outputExtras) = processLayer layer startingPosition currentValue layerExtra
      liftIO $ writeIORef value output
      --putStrLn $ "output = " ++ format output
      return outputExtras

  outputLayer <- liftIO $ readIORef value

  let normalizedOutputLayer = meanNorm outputLayer

  let normalizedOutputLayer2 = replicateVector (norm model) (width normalizedOutputLayer) `simpleElementMul` normalizedOutputLayer

  let output' = output model `matMul` normalizedOutputLayer2

  return (output', allExtras)

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

processLayer :: Layer -> Int -> Matrix -> HistoryKVs -> (Matrix, HistoryKVs)
processLayer layer startingPosition inputLayer extras =
  let normalized = meanNorm ( {-traceItem ("[" ++ show (layerNumber layer) ++ "] inputLayer") -} inputLayer)
      --normalized2 = map (zipWith (*) $ attention_norm layer) normalized
      normalized2 = replicateVector (attention_norm layer) (width normalized) `simpleElementMul` normalized
      (afterAttention, afterExtras) = selfAttention layer startingPosition normalized2 extras
      inputFF = --traceItem "inputFF" $
                afterAttention `matAdd` inputLayer -- normalized2
      outputFF = feedForward layer inputFF
      outputLayer = outputFF `matAdd` inputFF
  in (outputLayer, afterExtras)
  
instance Format [Matrix] where
  --format x = "(" ++ format (sum (join $ map join $ transpose $ map unMatrix x)) ++ ") head = " ++ format (head x)
  format x = "(" ++ format (sum (join $ map join $ map unMatrix x)) ++ ") head = " ++ format (head x)

selfAttention :: Layer -> Int -> Matrix -> HistoryKVs -> (Matrix, HistoryKVs)
selfAttention Layer{..} startingPosition inputSA (extraKCur, extraVCur) =
  let
      qCur = attention_wq `matMul` inputSA  -- [4x4096] = [??] * [??]
      kCur = attention_wk `matMul` inputSA
      vCur = attention_wv `matMul` inputSA
      headSize = height kCur `div` numberOfHeads
      kBlobs' :: [Matrix]
      kBlobs' = map Matrix $ transpose $ map (chunksOf headSize) $ unMatrix kCur
      kBlobs'' :: [Matrix]
      kBlobs'' = zipWith (\x y -> (matrixConcat [x, y])) extraKCur kBlobs'
--               (\(first:rest) -> (matrixConcat [traceItem "extraKCur" extraKCur, first]):rest) kBlobs'
      kBlobs :: [Matrix]
      kBlobs = map (Matrix . embedPositions 0 . unMatrix) kBlobs''
      qBlobs :: [Matrix]
      qBlobs = map (Matrix . embedPositions startingPosition) $ transpose $ map (chunksOf headSize) $ unMatrix qCur
      vBlobs' :: [Matrix]
      vBlobs' = map Matrix $ transpose $ map (chunksOf headSize) $ unMatrix vCur
      vBlobs :: [Matrix]
      vBlobs = zipWith (\x y -> (matrixConcat [x, y])) extraVCur vBlobs'
--               (\(first:rest) -> (matrixConcat [traceItem "extraVCur" extraVCur, first]):rest) vBlobs'


--      vBlobs :: [Matrix]
--      vBlobs = map Matrix $ transpose $ map (chunksOf headSize) $ unMatrix vCur
      kqs :: [Matrix]
      kqs = zipWith matMul kBlobs qBlobs -- 32 times: [4 x 4] = [4 x 128] * [4 x 128]
      kqs_scaled :: [Matrix]
      kqs_scaled = --map (matrixMap (/sqrt 128.0)) kqs
                   map (matrixMap (* (0x1.6a09e6p-4))) kqs
      kqs_masked :: [Matrix]
      kqs_masked = map (filterUpperDiagonal startingPosition) kqs_scaled
      kqs_softmax :: [Matrix]
      kqs_softmax = map (buildMatrixFromRows . map softMax . matrixRows) kqs_masked
      kqv :: [Matrix]
      kqv = zipWith matMul (map transposeMatrix vBlobs) kqs_softmax -- 32 times: [128 x 4] = [4 x 128] * [4 x 4]
      kqv_smashed = transposeMatrix (matrixConcat (map transposeMatrix kqv)) -- [??] = [4 x 4096] * [4096 x 4]
      output = attention_wo `matMul` kqv_smashed
  in (output, (kBlobs'', vBlobs))

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

filterUpperDiagonal :: Int -> Matrix -> Matrix
filterUpperDiagonal startingPosition (Matrix theMatrix) = Matrix $ map (\(theRow, i) -> filterAfter (startingPosition + i) theRow) $ zip theMatrix [1..]
filterUpperDiagonal _ (QuantizedMatrix _) = error "filterUpperDiagonal not defined for QuantizedMatrix"


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
matMul x y | height x /= height y = error $ "matrix heights don't match:\n" ++
             "x size is: [" ++ show (height x) ++ " x " ++ show (width x) ++ "]\n" ++
             "y size is: [" ++ show (height y) ++ " x " ++ show (width y) ++ "]\n" ++
             show (height x) ++ " /= " ++ show (height y)
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
