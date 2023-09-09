
module Rope (
  embedPositions
  ) where

import Data.List.Split

embedPositions :: Int -> [[Float]] -> [[Float]]
embedPositions startingPosition theArray =
  for (zip theArray [startingPosition..]) $ \(theColumn, position) ->
    unpairTuples $
    for (zip (pairTuples theColumn) $ map indexToTheta [0, 2..]) $ \(theValueTuple, theta) ->
      embedPosition theta position theValueTuple
  where
    for = flip map
    indexToTheta :: Int -> Double
    indexToTheta index = 10000.0 ** (-(fromIntegral index)/fromIntegral (length $ head theArray))
    pairTuples :: [a] -> [(a, a)]
    pairTuples = map listTo2Tuple . chunksOf 2
      where listTo2Tuple items =
              case items of
                [x, y] -> (x, y)
                _ -> error "pairTuples called with an odd number of items"
    unpairTuples :: [(a, a)] -> [a]
    unpairTuples = concat . map (\(x, y) -> [x, y])

embedPosition :: Double -> Int -> (Float, Float) -> (Float, Float)
embedPosition theta position (x, y) =
  (
    realToFrac $ x_double * cosalpha - y_double * sinalpha,
    realToFrac $ x_double * sinalpha + y_double * cosalpha
  )
  where alpha = fromIntegral position * theta
        cosalpha :: Double
        cosalpha = cos alpha
        sinalpha :: Double
        sinalpha = sin alpha
        x_double :: Double
        x_double = realToFrac x
        y_double :: Double
        y_double = realToFrac y
