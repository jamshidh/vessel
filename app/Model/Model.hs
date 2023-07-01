{-# LANGUAGE RecordWildCards #-}

module Model.Model (
  Layer(..),
  Model(..),
  rawModelToModel
  ) where

import Control.Monad
import Data.Binary
import Data.Binary.Get
import qualified Data.ByteString.Char8 as BC
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Maybe

import Format

import Model.Int
import Model.Tensor
import Model.Token


data RawModel =
  RawModel {
    first :: String,
    n_embd :: Int32le,
    n_mult :: Int32le,
    n_head :: Int32le,
    n_layer :: Int32le,
    n_rot :: Int32le,
    f16 :: Int32le,
    tokens' :: [Token],
    tensors :: Map String Tensor
  } deriving (Show)


instance Binary RawModel where
  get = do
    f <- getByteString 4
    n_vocab' <- getInt32le
    e <- get
    m <- get
    h <- get
    l <- get
    r <- get
    f16' <- get
    tokens <- replicateM (fromIntegral n_vocab') get
    tensors' <- fmap (Map.fromList) $ replicateM 291 $ fmap (\t -> (name t, t)) get
    return $ RawModel (BC.unpack f) e m h l r f16' tokens tensors'
  put = undefined

instance Format RawModel where
  format RawModel{..} =
    "Model: \n"
    ++ "  first = " ++ show first ++ "\n"
    ++ "  n_embd = " ++ show n_embd ++ "\n"
    ++ "  n_mult = " ++ show n_mult ++ "\n"
    ++ "  n_head = " ++ show n_head ++ "\n"
    ++ "  n_layer = " ++ show n_layer ++ "\n"
    ++ "  n_rot = " ++ show n_rot ++ "\n"
    ++ "  f16 = " ++ show f16 ++ "\n"
    ++ "  tokens: " ++ unwords (take 1000 $ map format tokens') ++ "\n"
    ++ "  tensor: " ++ unlines (map (format . snd) $ Map.toList tensors) ++ "\n"


data Model = Model {
  tokens :: [Token],
  layers :: [Layer],
  norm :: [Float],
  output :: [[Float]],
  tokenEmbeddings :: [[Float]]
  }

data Layer =
  Layer {
    layerNumber :: Int,
    attention_wk :: [[Float]],
    attention_wo :: [[Float]],
    attention_wq :: [[Float]],
    attention_wv :: [[Float]],
    attention_norm :: [Float],
    feed_forward_w1 :: [[Float]],
    feed_forward_w2 :: [[Float]],
    feed_forward_w3 :: [[Float]],
    ffn_norm :: [Float]
  }

rawModelToModel :: RawModel -> Model
rawModelToModel RawModel{..} =
  Model {
    tokens = tokens',
    layers = map (getLayer tensors) [0..fromIntegral (n_layer-1)],
    norm = error "norm undefined",
    output = error "output undefined",
    tokenEmbeddings = getFloatArray tensors "tok_embeddings.weight"
  }

getLayer :: Map String Tensor -> Int -> Layer
getLayer t i = Layer {
    layerNumber = i,
    attention_wk = getFloatArray t $ "layers." ++ show i ++ ".attention.wk.weight",
    attention_wo = getFloatArray t $ "layers." ++ show i ++ ".attention.wo.weight",
    attention_wq =  getFloatArray t $ "layers." ++ show i ++ ".attention.wq.weight",
    --attention_wq = error $ show (fmap (B16.encode . BC.take 100 . elems) $ Map.lookup "layers.0.attention.wq.weight" t),
    attention_wv = getFloatArray t $ "layers." ++ show i ++ ".attention.wv.weight",
    attention_norm = getFloatList t $ "layers." ++ show i ++ ".attention_norm.weight",
    feed_forward_w1 = getFloatArray t $ "layers." ++ show i ++ ".feed_forward.w1.weight",
    feed_forward_w2 = getFloatArray t $ "layers." ++ show i ++ ".feed_forward.w2.weight",
    feed_forward_w3 = getFloatArray t $ "layers." ++ show i ++ ".feed_forward.w3.weight",
    ffn_norm = getFloatList t $ "layers." ++ show i ++ ".ffn_norm.weight"
    }


getFloatList :: Map String Tensor -> String -> [Float]
getFloatList t name = tensorToFloatList $ fromMaybe (error $ show name ++ " undefined in the model tensors: " ++ show (Map.keys t)) $ Map.lookup name t

getFloatArray :: Map String Tensor -> String -> [[Float]]
getFloatArray t name = tensorToFloatArray $ fromMaybe (error $ show name ++ " undefined in the model tensors: " ++ show (Map.keys t)) $ Map.lookup name t
