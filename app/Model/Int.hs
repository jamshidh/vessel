{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Model.Int (
  Int32le
  ) where

import Data.Binary
import Data.Binary.Get
import Data.Binary.Put
import Data.Int

newtype Int32le = Int32le Int32 deriving (Eq, Enum, Ord, Num, Real, Integral)

instance Show Int32le where
  show (Int32le v) = show v

instance Binary Int32le where
  get = fmap Int32le getInt32le
  put (Int32le v) = putInt32le v
  

