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
import Data.ByteString (ByteString)
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Lazy as BL
import Data.IORef
import Data.List (sortOn, transpose)
import Data.List.Split
import qualified Data.Vector.Storable as V
import Foreign.C.String
import Numeric.Half
import System.IO
import System.IO.Unsafe

import Converse

import Format

import Model.Float ()
import Model.Matrix
import Model.Model
import Model.Tensor
import Model.Token
import Model.Vector
import Rope

--import Debug.Trace

--traceItem :: Format a => String -> a -> a
--traceItem name item = trace (name ++ ": " ++ format item) item

numberOfHeads :: Int
numberOfHeads = 32


type OutcomeSpace = [(Int, Float)]

main :: IO ()
main = do
  x <- BL.readFile "ggml-alpaca-7b-q4.bin"
  let rawModel = decode x
  let model :: Model
      model = rawModelToModel rawModel

  --putStrLn $ model `deepseq` "abcd"
  --putStrLn $ format rawModel

  _ <-
    runConverse model $ forever $ do
      input <- prompt "> "
      _ <- addNewTokens " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
      outcomes <- addNewTokens $ BC.pack $ "### Instruction:\n\n" ++ input ++ "\n### Response:\n\n"
      outcomes2 <- getAndPrintFullResponse outcomes
      printTopTokens outcomes2

  return ()


prompt :: MonadIO m => String -> m String
prompt thePrompt = do
  liftIO $ putStr thePrompt
  liftIO $ hFlush stdout
  liftIO getLine

getAndPrintFullResponse :: OutcomeSpace -> Converse OutcomeSpace
getAndPrintFullResponse outcomes = do
  let nextToken = getNextToken outcomes
  if nextToken == 2
    then do
    liftIO $ putStrLn ""
    return outcomes
    else do
    printToken nextToken
    liftIO $ hFlush stdout
    nextOutcomes <- handleNewTokens [nextToken]
    getAndPrintFullResponse nextOutcomes --thank you sir, may I have another

getNextToken :: OutcomeSpace -> Int
getNextToken = fst . head --just take the most probable for now

addNewTokens :: ByteString -> Converse OutcomeSpace
addNewTokens phrase = do
  (model, _, _) <- State.get
  fmap last $
    forM (chunksOf 9 (1:tokenize (tokenTrie model) phrase)) $ \phraseChunk -> do
      handleNewTokens phraseChunk



handleNewTokens :: [Int] -> Converse OutcomeSpace
handleNewTokens phrase = do
  (model, tokenHistory, historyKVs) <- State.get
  let position = length tokenHistory - 1
  (Matrix output, extras) <- runNN model position historyKVs phrase

  let tokensWithProbs = logitsToTopProbabilities (tokenHistory ++ phrase) $ last output

  State.put (model, tokenHistory ++ phrase, extras)
  
  return tokensWithProbs


scale :: [Int] -> (Int, Float) -> Float
scale tokenHistoryLast (t, l) =
  let scaleFactor =
        case (t `elem` tokenHistoryLast, l<0.0) of
          (False, _) -> 10
          (_, True) -> 10*1.3
          (_, False) -> 10/1.3
  in l*scaleFactor

logitsToTopProbabilities :: [Int] -> [Float] -> OutcomeSpace
logitsToTopProbabilities tokenHistory logits= 
  let tokens' = zip [0..] logits
      tokensScaled = map (\(t, l) -> (t, scale tokenHistoryLast (t, l))) tokens'
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

printTopTokens :: OutcomeSpace -> Converse ()
printTopTokens tokensWithProbs = do
  (model, _, _) <- State.get
  forM_ tokensWithProbs $ \(tokenInt, prob) ->
    liftIO $ putStrLn $ format prob ++ ": (" ++ show tokenInt ++ ") " ++ show (format $ tokens model !! tokenInt)

--instance Format [Int] where
--  format = show

runNN :: Model -> Int -> [HistoryKVs] -> [Int] -> Converse (Matrix, [HistoryKVs])
runNN model startingPosition extras embd = do
  let inputLayer = Matrix $ map (getRow $ tokenEmbeddings model) embd

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

matrixVectors :: Matrix -> [Vector]
matrixVectors (Matrix vectors) = vectors
matrixVectors _ = error "matrixVectors not defined for QuantizedMatrix"




qMatrixVectors :: Matrix -> [QuantizedVector]
qMatrixVectors (QuantizedMatrix matrixData) = matrixData
qMatrixVectors _ = error "trying to convert non quantized Matrix to quantized vectors"


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
      kqs_softmax = map (Matrix . map softMax . matrixVectors) kqs_masked
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

matrixConcat :: [Matrix] -> Matrix
matrixConcat matrixList = Matrix $ concat $ map unMatrix matrixList


silu :: Float -> Float
silu x =
  let x_double :: Double
      x_double = realToFrac x
  in realToFrac $ x_double/(1+exp (-x_double))

replicateVector :: Vector -> Int -> Matrix
replicateVector vals i = Matrix $ replicate i vals

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
softMax theRow = normalize . map (exp' . (\v -> v - maximum theRow)) $ theRow

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
  Matrix $ map (\yRow -> map (\xCol -> xCol `qdot` yRow) $ qMatrixVectors x) $ qMatrixVectors $ quantizeMatrix y
matMul _ _ = error "unsupported case called in matMul"

slowDot :: [Float] -> [Float] -> Float
--slowDot x y = realToFrac $ sum $ zipWith (*) (map realToFrac x) (map realToFrac y :: [Double])
--slowDot x y = sum $ zipWith (*) x y

slowDot x y = zipFold (\v1 v2 s -> fusionMultiplySumAllFloat v1 v2 s) (0.0::Float) x y

dot :: Vector -> Vector -> Float
--dot x y | trace ("dot, length x = " ++ show (length x) ++ ", length y = " ++ show (length y)) False = undefined
dot x y | length x /= length y = error $ "dot product lengths do not match: " ++ show (length x) ++ "/=" ++ show (length y)
dot x y = realToFrac $ zipFold fusionMultiplySum (0.0::Double) x y
  --sum $ zipWith (*) (map realToFrac x::[Double]) (map realToFrac y::[Double])



qdot :: QuantizedVector -> QuantizedVector -> Float
qdot (QuantizedVector x) (QuantizedVector y) = x `quantized_vector_dot` y





zipFold :: (a -> b -> c -> c) -> c -> [a] -> [b] -> c
zipFold _ initVal [] [] = initVal
zipFold f initVal (firstx:restx) (firsty:resty) =
  let newVal = f firstx firsty initVal
  in zipFold f newVal restx resty
zipFold _ _ _ _ = error "mismatched array sizes in call to zipFold"

foreign import ccall "fusionMultiplySum" fusionMultiplySum :: Float -> Float -> Double -> Double
foreign import ccall "fusionMultiplySumAllFloat" fusionMultiplySumAllFloat :: Float -> Float -> Float -> Float

foreign import ccall "vector_dot" vector_dot :: Int -> CString -> CString -> Float

quantized_vector_dot :: UnpackedQuantizedVector -> UnpackedQuantizedVector -> Float
--quantized_block_dot (QuantizedBlock f1 n1) (QuantizedBlock f2 n2) | trace ("f1 = " ++ format f1 ++ ", f2 = " ++ format f2 ++ ", n1 = " ++ show n1 ++ ", n2 = " ++ show n2 ++ ", int dot = " ++ show (sum $ zipWith (*) n1 n2)) False = undefined
quantized_vector_dot v1 v2 | V.length v1 /= V.length v2 = error "vector lengths different in call to quantized_vector_dot"
quantized_vector_dot v1 v2 =
  let bytes1 = unpackedQuantizedVectorToByteString v1
      bytes2 = unpackedQuantizedVectorToByteString v2
  in unsafePerformIO $
  B.useAsCStringLen bytes1 $ \(p1, _) ->
  B.useAsCStringLen bytes2 $ \(p2, _) ->
                      return $ vector_dot (V.length v1) p1 p2


{-
--quantized_block_dot (QuantizedBlock f1 ints1) (QuantizedBlock f2 ints2) = --traceItem "quantized_block_dot" $ 
--  f1 * f2 * (fromIntegral $ sum $ zipWith (*) ints1 ints2)
  let f1 = V.unsafeCast $ castPtr p1 :: Float
      f2 = V.unsafeCast $ castPtr p2 :: Float
  in f1 * f2 * ints1 `dot_Int4X32` ints2
-}

meanNorm :: Matrix -> Matrix
meanNorm m@(Matrix theFloats) = 
  let squaresSummed = zipWith ((\v1 v2 -> zipFold fusionMultiplySum (0.0::Double) v1 v2)) theFloats theFloats
      mean = map (/realToFrac (height m)) squaresSummed :: [Double]
      scaleFactors = map realToFrac $ map (1/) $ (map (sqrt . (+1e-5)) mean) :: [Float]
  in Matrix $ zipWith (\sf f -> map (sf*) f) scaleFactors theFloats
meanNorm (QuantizedMatrix _) = error "meanNorm not defined for QuantizedMatrix"
