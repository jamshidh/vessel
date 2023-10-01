
module Converse where

import Control.Monad.Trans.State

import Model.Tensor

type HistoryKVs = ([Matrix], [Matrix])



type Converse a = StateT ([Int], [HistoryKVs]) IO a

runConverse = flip runStateT (emptyTokenHistory, emptyHistory)


emptyHistory = (replicate 32 (replicate 32 (Matrix []), replicate 32 (Matrix [])))
emptyTokenHistory = [0::Int]
