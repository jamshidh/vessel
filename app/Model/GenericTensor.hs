{-# LANGUAGE RecordWildCards #-}

module Model.GenericTensor (
  GenericTensor(..),
  TensorType(..)
  ) where

import Control.Monad
import Data.Binary
import Data.Binary.Get
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BC
import Data.Int
import Data.List (intercalate)

import Format

data TensorType =
  Q4_0
  | F32 deriving (Show)

instance Binary TensorType where
  get = do
    v <- getInt32le
    case v of
      0 -> return F32
      2 -> return Q4_0
      _ -> error $ "bad tensor type number: " ++ show v
  put = undefined
  
typeSize :: TensorType -> Int
typeSize Q4_0 = 20
typeSize F32 = 4
--typeSize v = error $ "unknown numberElements: " ++ show v

blockSize :: TensorType -> Int
blockSize Q4_0 = 32
blockSize F32 = 1
--blockSize v = error $ "unknown blockSize: " ++ show v




data GenericTensor =
  GenericTensor {
    fType :: TensorType,
    dim_num_elems :: [Int32],
    name :: String,
--    elems :: [Int32le]
    elems :: BC.ByteString
  } deriving (Show)

instance Format GenericTensor where
  format GenericTensor{..} =
    "Tensor (" ++ show fType ++ "): "
    ++ "[" ++ intercalate " ⨯ " (map show dim_num_elems) ++ "] "
    ++ name ++ "\n"
    ++ "  elems = " ++ show (B16.encode $ BC.take 100 elems)



instance Binary GenericTensor where
  get = do
    n_dims' <- getInt32le
    len' <- getInt32le
    ft <- get
    dim_num_elems' <- replicateM (fromIntegral n_dims') getInt32le
    name' <- getByteString (fromIntegral len')
--    elems' <- replicateM (fromIntegral (product dim_num_elems')) get
    elems' <- getByteString (typeSize ft * fromIntegral (product dim_num_elems')`div` blockSize ft)
    return $ GenericTensor ft dim_num_elems' (BC.unpack name') elems'
  put = undefined

