
module Model.Tensor (
  getRow
  ) where

import qualified Data.Vector.Storable as V

import Model.Float ()
import Model.Int4X32
import Model.Matrix
import Model.Vector

--import Debug.Trace

quantizedBlockToFloats :: QuantizedBlock -> [Float]
quantizedBlockToFloats (QuantizedBlock theFloat theNibbles) = 
  map (* theFloat) $ map fromIntegral $ unpackInt4X32 theNibbles


getRow :: Matrix -> Int -> Vector
--getRow m i | trace ("getRow: " ++ format m ++ " " ++ show i) False = undefined
getRow (QuantizedMatrix matrixData) i = concat . map quantizedBlockToFloats . V.toList . (\(QuantizedVector v) -> v) . (matrixData !!) $ i -- concat . map blockToFloats . splitIntoBlocks . bytesForRow m
getRow _ _ = error "getRow not definted for non-quantized Matrix"

