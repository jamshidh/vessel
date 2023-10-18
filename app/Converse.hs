
module Converse (
  Converse,
  HistoryKVs,
  runConverse,
  printToken
  ) where

import Control.Monad.IO.Class
import Control.Monad.Trans.State

import Model.Matrix
import Model.Model

import Format

type HistoryKVs = ([Matrix], [Matrix])



type Converse a = StateT (Model, [Int], [HistoryKVs]) IO a

runConverse :: Model -> Converse a -> IO (a, (Model, [Int], [HistoryKVs]))
runConverse model = flip runStateT (model, emptyTokenHistory, emptyHistory)

emptyHistory :: [HistoryKVs]
emptyHistory = (replicate 32 (replicate 32 (Matrix []), replicate 32 (Matrix [])))

emptyTokenHistory :: [Int]
emptyTokenHistory = [0::Int]

printToken :: Int -> Converse ()
printToken theToken = do
  (model, _, _) <- get
  liftIO $ putStr $ format (tokens model !! theToken)
