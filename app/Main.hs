{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import Control.Monad
import Data.Binary
import qualified Data.ByteString.Lazy as BL
import Data.Int
import Data.IORef
import Data.List (transpose)
import Data.List.Split

import Format

import Model.Float ()
import Model.Model
--import Model.Tensor
--import Model.Token

import Rope

import Debug.Trace

numberOfHeads :: Int
numberOfHeads = 32


main :: IO ()
main = do
  x <- BL.readFile "ggml-alpaca-7b-q4.bin"
  let model :: Model
      model = rawModelToModel $ decode x

{-
  let phrase = " Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"

  let theTrie = tokensToTokenTrie $ tokens model

  let phraseTokens = 1:tokenize theTrie phrase
  
  let phraseTokensFromCPP = [1, 13866, 338, 385, 15278, 393, 16612, 263, 3414, 29889, 14350, 263, 2933, 393, 8210, 368, 4866, 29879, 278, 2009, 29889, 13, 13]

  putStrLn $ "phraseTokens: " ++ show phraseTokens
  putStrLn $ "phraseTokensFromCPP: " ++ show phraseTokensFromCPP
  
  putStrLn $ show $ map (\i -> (i, format $ tokens model !! i)) phraseTokens
  putStrLn $ show $ map (\i -> (i, format $ tokens model !! i)) phraseTokensFromCPP
-}


  
--  putStrLn $ format model

  let embd = [0,1,2,3]

  let inputLayer = map (tokenEmbeddings model !!) embd

  value <- newIORef inputLayer
  
  putStrLn $ "inputLayer = " ++ format inputLayer

--  let outputLayer = applyPipeline (map processLayer $ reverse $ layers model) inputLayer

  writeIORef value inputLayer


  forM_ (layers model) $ \layer -> do
    currentValue <- readIORef value
    let output = processLayer layer currentValue
    writeIORef value output
    putStrLn $ "output = " ++ format output


  outputLayer <- readIORef value

  putStrLn $ "outputLayer = " ++ format outputLayer

  let normalizedOutputLayer = meanNorm outputLayer

  putStrLn $ "normalizedOutputLayer = " ++ format normalizedOutputLayer

{-  
  let normalizedOutputLayer2 = map (zipWith (*) $ norm model) normalizedOutputLayer

  putStrLn $ "normalizedOutputLayer2 = " ++ format normalizedOutputLayer2

  let output' = output model `matMul` normalizedOutputLayer2

  putStrLn $ "output' = " ++ format output'
-}


  

{-
applyPipeline :: [(a->a)] -> a -> a
applyPipeline = flip $ foldr ($)
-}

processLayer :: Layer -> [[Float]] -> [[Float]]
processLayer layer inputLayer =
  let normalized = meanNorm inputLayer
      normalized2 = map (zipWith (*) $ attention_norm layer) normalized
      afterAttention = selfAttention layer normalized2
      inputFF = afterAttention `matAdd` normalized2
      outputFF = feedForward layer inputFF
      outputLayer = outputFF `matAdd` inputFF
  in trace (
    "afterAttention = " ++ format afterAttention ++ "\n" ++
    "inpFF = " ++ format inputFF
    )
     outputLayer
  

selfAttention :: Layer -> [[Float]] -> [[Float]]
selfAttention Layer{..} inputSA =
  let
      qCur = attention_wq `qMatMul` inputSA  -- [4x4096] = [??] * [??]
      kCur = attention_wk `qMatMul` inputSA
      vCur = attention_wv `qMatMul` inputSA
      headSize = (length $ head kCur) `div` numberOfHeads
      kBlobs = map embedPositions $ transpose $ map (chunksOf headSize) kCur
      qBlobs = map embedPositions $ transpose $ map (chunksOf headSize) qCur
      vBlobs = map embedPositions $ transpose $ map (chunksOf headSize) vCur
      kqs = zipWith matMul kBlobs qBlobs -- 32 times: [4 x 4] = [4 x 128] * [4 x 128]
      kqs_scaled = map (map (map (/sqrt 128.0))) kqs
      kqs_masked = map filterUpperDiagonal kqs_scaled
      kqs_softmax = map (map softMax) kqs_masked
      kqv = zipWith matMul (map transpose vBlobs) kqs_softmax -- 32 times: [128 x 4] = [4 x 128] * [4 x 4]
      output = attention_wo `qMatMul` transpose (concat (map transpose kqv)) -- [??] = [4 x 4096] * [4096 x 4]
  in trace (
    "==========================\nattention_wk(" ++ show layerNumber ++ "): " ++ format attention_wk ++ "\n" ++
    "kCur = " ++ format kCur ++ "\n" ++
    "qCur = " ++ format qCur ++ "\n" ++
    "vCur = " ++ format vCur ++ "\n" ++
    "first mul: " ++ format (head kqs) ++ "\n" ++
    "first mul softmax: " ++ format (head kqs_softmax) ++ "\n" ++
    "first kqv: " ++ format (head kqv)
    )
    output

feedForward :: Layer -> [[Float]] -> [[Float]]
feedForward layer inpFF = 
  let cur1 = meanNorm inpFF
      cur2 = map (zipWith (*) (ffn_norm layer)) cur1
      tmp = feed_forward_w3 layer `qMatMul` cur2
      cur3 = feed_forward_w1 layer `qMatMul` cur2
      cur4 = map (map silu) cur3
      cur5 = zipWith (zipWith (*)) cur4 tmp
      cur6 = feed_forward_w2 layer `qMatMul` cur5
  in cur6




silu :: Float -> Float
silu x = 1/(1+exp (-x))








softMax :: [Float] -> [Float]
softMax theRow = normalize . map (exp . (\v -> v - maximum theRow)) $ theRow

filterUpperDiagonal :: [[Float]] -> [[Float]]
filterUpperDiagonal theMatrix = map (\(theRow, i) -> filterAfter i theRow) $ zip theMatrix [1..]

filterAfter :: Int -> [Float] -> [Float]
filterAfter i theRow = take i theRow ++ replicate (length theRow - i) (-inf)
  where inf = 1/0

normalize :: [Float] -> [Float]
normalize values = map (/sum values) values


matAdd :: [[Float]] -> [[Float]] -> [[Float]]
matAdd = zipWith vectAdd

vectAdd :: [Float] -> [Float] -> [Float]
vectAdd = zipWith (+)

qMatMul :: [[Float]] -> [[Float]] -> [[Float]]
qMatMul x y =
  map (\yRow -> map (\xCol -> xCol `quantize_dot` yRow) x) y

matMul :: [[Float]] -> [[Float]] -> [[Float]]
matMul x y =
  map (\yRow -> map (\xCol -> xCol `dot` yRow) x) y

dot :: [Float] -> [Float] -> Float
dot x y | length x /= length y = error $ "dot product lengths do not match: " ++ show (length x) ++ "/=" ++ show (length y)
dot x y = sum $ zipWith (*) x y


quantize_dot :: [Float] -> [Float] -> Float
quantize_dot x y = sum $ zipWith (quantized_block_dot) (quantize y) (quantize x)


quantized_block_dot :: QuantizedBlock -> QuantizedBlock -> Float
quantized_block_dot (QuantizedBlock f1 ints1) (QuantizedBlock f2 ints2) =
  f1 * f2 * (fromIntegral $ sum $ zipWith (*) ints1 ints2)
  
quantize :: [Float] -> [QuantizedBlock]
quantize [] = []
quantize floats =
  let (firstBlockFloats, rest) = splitAt 32 floats
  in quantize_single_block firstBlockFloats:quantize rest

quantize_single_block :: [Float] -> QuantizedBlock
quantize_single_block floats | length floats /= 32 = error $ "quantization blocks must have length 32, actually is " ++ show (length floats)
quantize_single_block floats =
  QuantizedBlock d (map (round . (inverse_d *)) floats)
  where d = maximum (map abs floats)/7
        inverse_d = 1/d

data QuantizedBlock = QuantizedBlock Float [Int16] deriving (Show)



meanNorm :: [[Float]] -> [[Float]]
meanNorm theFloats = 
  let scaleFactors = map ((1/) . sqrt . (+1e-5) . (/4096) . sum . map (\val -> val*val)) $ theFloats
  in zipWith (\sf f -> map (sf*) f) scaleFactors theFloats
