{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module Model.Token (
  Token,
  tokenize,
  tokensToTokenTrie
  ) where

import Control.DeepSeq
import Data.Binary
import Data.Binary.Get
import qualified Data.ByteString.Char8 as BC
import Data.Trie (Trie)
import qualified Data.Trie as Trie
import GHC.Generics

import Format

newtype Token = Token String deriving (Show, Generic, NFData)

instance Format Token where
  format (Token t) = t



instance Binary Token where
  get = do
    size <- getInt32le
    tokenString <- getByteString $ fromIntegral size
    return $ Token $ BC.unpack tokenString
  put = undefined


--temporary hack to get tokenization to match the c version.  I'm not sure why these are reversed from c, I'll have to dig in more later to understand if I'm doing something wrong
orderHack :: [Int] -> [Int]
orderHack [] = []
orderHack (2277:29937:rest) = 29937:2277:orderHack rest
orderHack (x:rest) = x:orderHack rest

tokenize :: Trie Int -> BC.ByteString -> [Int]
tokenize theTrie = orderHack . tokenize' theTrie

tokenize' :: Trie Int -> BC.ByteString -> [Int]
tokenize' _ "" = []
tokenize' theTrie s =
  case Trie.match theTrie s of
    Nothing -> error $ "can't find token to match to: " ++ show s
    Just (_, i, rest) -> i:tokenize' theTrie rest

tokensToTokenTrie :: [Token] -> Trie Int
tokensToTokenTrie tokens = Trie.fromList $ reverse $ zip (map tokenAsByteString tokens) [0..]
  where tokenAsByteString (Token s) = BC.pack s


