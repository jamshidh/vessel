{-# LANGUAGE FlexibleInstances #-}
{-# OPTIONS -fno-warn-orphans #-}

module Model.Float (
  formatFloatBytes
  ) where

import qualified Data.ByteString as B
import qualified Data.ByteString.Base16 as B16
import qualified Data.ByteString.Char8 as BC
import Data.List (intercalate)

import qualified Data.Vector.Storable as V
import Numeric

import Format

instance Format [Float] where
  format x = "[" ++ show (length x) ++ "]\n["
             ++ intercalate ", " (map format (take 10 x))
             ++ "]"

instance Format [[Float]] where
  format x = "[" ++ show (length x) ++ " x " ++ show (length $ head x) ++ "]\n"
--             ++ unlines (map (("    " ++) . show . take 5) (take 5 x))
             ++ unlines (map showLine (take 5 x))
             ++ (if length x > 5 then "    ....(etc)" else "")
    where showLine v = (++ (if length v > 5 then " | ...." else "")) . ("    " ++) . intercalate " | " . map format . take 5 $ v

instance Format Float where
  format f =
    widen 15 ((if (f >= 0) then " " else "") ++ showHFloat f "")
    where widen i s = s ++ replicate (i-length s) ' '

  
  {-
  format f =
    widen 13 ((if (f >= 0) then " " else "") ++ floatString)

    where floatString = show f
          widen i s = s ++ replicate (i-length s) ' '
-}



instance Format [Double] where
  format x = "[" ++ show (length x) ++ "]\n["
             ++ intercalate ", " (map format (take 10 x))
             ++ "]"

instance Format [[Double]] where
  format x = "[" ++ show (length x) ++ " x " ++ show (length $ head x) ++ "]\n"
--             ++ unlines (map (("    " ++) . show . take 5) (take 5 x))
             ++ unlines (map showLine (take 5 x))
             ++ (if length x > 5 then "    ....(etc)" else "")
    where showLine v = (++ (if length v > 5 then " | ...." else "")) . ("    " ++) . intercalate " | " . map format . take 5 $ v

instance Format Double where
  format f = showHFloat f ""

{-
formatFloat :: Float -> String
formatFloat = widen 13 . printf "%1.7e"
    where widen i s = s ++ replicate (i-length s) ' '
-}

formatFloatBytes :: Float -> String
formatFloatBytes f = BC.unpack $ (B16.encode $ B.pack $ V.toList $ V.unsafeCast $ V.singleton f)
