
module Model.Tensor (
  getRow
  ) where

import Model.Float ()
import Model.Matrix
import Model.Vector

--import Debug.Trace

getRow :: Matrix -> Int -> Vector
--getRow m i | trace ("getRow: " ++ format m ++ " " ++ show i) False = undefined
getRow (QuantizedMatrix matrixData) i = dequantize $ matrixData !!  i 
getRow _ _ = error "getRow not definted for non-quantized Matrix"

