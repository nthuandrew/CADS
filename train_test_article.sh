#!/bin/bash

for fold in {1..5}
do
    # echo "Running main_article.py for fold $fold"
    # python main_article.py $fold

    echo "Running main_article_test.py for fold $fold"
    python main_article_test.py $fold
done
