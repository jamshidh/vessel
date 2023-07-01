
module Rope where

import Data.List.Split

embedPositions :: [[Float]] -> [[Float]]
embedPositions theArray =
  for (zip theArray [0..]) $ \(theColumn, position) ->
    unpairTuples $
    for (zip (pairTuples theColumn) $ map indexToTheta [0, 2..]) $ \(theValueTuple, theta) ->
      embedPosition theta position theValueTuple
  where
    for = flip map
    indexToTheta :: Int -> Float
    indexToTheta index = 10000.0 ** (-(fromIntegral index::Float)/fromIntegral (length $ head theArray))
    pairTuples :: [a] -> [(a, a)]
    pairTuples = map (\[x, y] -> (x, y)) . chunksOf 2
    unpairTuples :: [(a, a)] -> [a]
    unpairTuples = concat . map (\(x, y) -> [x, y])

embedPosition :: Float -> Int -> (Float, Float) -> (Float, Float)
embedPosition theta position (x, y) =
  (
    x * cos alpha - y * sin alpha,
    x * sin alpha + y * cos alpha
  )
  where alpha = fromIntegral position * theta
