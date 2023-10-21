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
import qualified Data.ByteString.Char8 as BC
import qualified Data.ByteString.Lazy as BL
import Data.IORef
import Data.List (sortOn, transpose)
import Data.List.Split
import Numeric.Half
import System.IO

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
      kBlobs' = map Matrix $ transpose $ map (chunksOf headSize) $ matrixVectors kCur
      kBlobs'' :: [Matrix]
      kBlobs'' = zipWith (\x y -> (matrixConcat [x, y])) extraKCur kBlobs'
--               (\(first:rest) -> (matrixConcat [traceItem "extraKCur" extraKCur, first]):rest) kBlobs'
      kBlobs :: [Matrix]
      kBlobs = map (Matrix . embedPositions 0 . matrixVectors) kBlobs''
      qBlobs :: [Matrix]
      qBlobs = map (Matrix . embedPositions startingPosition) $ transpose $ map (chunksOf headSize) $ matrixVectors qCur
      vBlobs' :: [Matrix]
      vBlobs' = map Matrix $ transpose $ map (chunksOf headSize) $ matrixVectors vCur
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
matrixConcat matrixList = Matrix $ concat $ map matrixVectors matrixList


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

meanNorm :: Matrix -> Matrix
meanNorm m@(Matrix theFloats) = 
  let squaresSummed = zipWith ((\v1 v2 -> zipFold fusionMultiplySum (0.0::Double) v1 v2)) theFloats theFloats
      mean = map (/realToFrac (height m)) squaresSummed :: [Double]
      scaleFactors = map realToFrac $ map (1/) $ (map (sqrt . (+1e-5)) mean) :: [Float]
  in Matrix $ zipWith (\sf f -> map (sf*) f) scaleFactors theFloats
meanNorm (QuantizedMatrix _) = error "meanNorm not defined for QuantizedMatrix"
