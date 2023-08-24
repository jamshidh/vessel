{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}

module Model.Model (
  Layer(..),
  Model(..),
  rawModelToModel
  ) where

import Control.Monad
import Control.DeepSeq
import Data.Binary
import Data.Binary.Get
import qualified Data.ByteString.Char8 as BC
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Maybe
import GHC.Generics

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
    tensors :: Map String GenericTensor
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
  norm :: Vector,
  output :: Matrix,
  tokenEmbeddings :: Matrix
  } deriving (Generic, NFData)

data Layer =
  Layer {
    layerNumber :: Int,
    attention_wk :: Matrix,
    attention_wo :: Matrix,
    attention_wq :: Matrix,
    attention_wv :: Matrix,
    attention_norm :: Vector,
    feed_forward_w1 :: Matrix,
    feed_forward_w2 :: Matrix,
    feed_forward_w3 :: Matrix,
    ffn_norm :: Vector
  } deriving (Generic, NFData)

rawModelToModel :: RawModel -> Model
rawModelToModel rawModel =
  Model {
    tokens = tokens' rawModel,
    layers = map (getLayer rawModel) [0..fromIntegral (n_layer rawModel-1)],
    --norm = Vector [], --dummy value for deepseq
    norm = getModelVector rawModel "norm.weight",
    --output = Matrix [[]], --dummy value for deepseq
    output = getModelMatrix rawModel "output.weight",
    tokenEmbeddings = getModelMatrix rawModel "tok_embeddings.weight"
  }

getLayer :: RawModel -> Int -> Layer
getLayer rawModel i = Layer {
    layerNumber = i,
    attention_wk = getModelMatrix rawModel $ "layers." ++ show i ++ ".attention.wk.weight",
    attention_wo = getModelMatrix rawModel $ "layers." ++ show i ++ ".attention.wo.weight",
    attention_wq =  getModelMatrix rawModel $ "layers." ++ show i ++ ".attention.wq.weight",
    --attention_wq = error $ show (fmap (B16.encode . BC.take 100 . elems) $ Map.lookup "layers.0.attention.wq.weight" t),
    attention_wv = getModelMatrix rawModel $ "layers." ++ show i ++ ".attention.wv.weight",
    attention_norm = getModelVector rawModel $ "layers." ++ show i ++ ".attention_norm.weight",
    feed_forward_w1 = getModelMatrix rawModel $ "layers." ++ show i ++ ".feed_forward.w1.weight",
    feed_forward_w2 = getModelMatrix rawModel $ "layers." ++ show i ++ ".feed_forward.w2.weight",
    feed_forward_w3 = getModelMatrix rawModel $ "layers." ++ show i ++ ".feed_forward.w3.weight",
    ffn_norm = getModelVector rawModel $ "layers." ++ show i ++ ".ffn_norm.weight"
    }


getModelVector :: RawModel -> String -> Vector
getModelVector RawModel{..} name = tensorToVector $ fromMaybe (error $ show name ++ " undefined in the model tensors: " ++ show (Map.keys tensors)) $ Map.lookup name tensors

getModelMatrix :: RawModel -> String -> Matrix
getModelMatrix RawModel{..} name = tensorToMatrix $ fromMaybe (error $ show name ++ " undefined in the model tensors: " ++ show (Map.keys tensors)) $ Map.lookup name tensors
