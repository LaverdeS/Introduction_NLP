python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers True --remove_special_characters True --remove_stopwords True --stem True

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rn_rsc_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rn_rsc_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rn_rsc_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rn_rsc_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rn_rsc_rsw_s.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers True --remove_special_characters True --remove_stopwords True --stem False

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rn_rsc_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rn_rsc_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rn_rsc_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rn_rsc_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rn_rsc_rsw.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers True --remove_special_characters False --remove_stopwords True --stem True

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rn_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rn_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rn_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rn_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rn_rsw_s.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers True --remove_special_characters True --remove_stopwords False --stem True

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rn_rsc_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rn_rsc_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rn_rsc_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rn_rsc_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rn_rsc_s.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers False --remove_special_characters True --remove_stopwords True --stem True

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rsc_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rsc_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rsc_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rsc_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rsc_rsw_s.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers True --remove_special_characters True --remove_stopwords False --stem False

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rn_rsc.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rn_rsc.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rn_rsc.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rn_rsc.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rn_rsc.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers True --remove_special_characters False --remove_stopwords True --stem False

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rn_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rn_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rn_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rn_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rn_rsw.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers True --remove_special_characters False --remove_stopwords False --stem True

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rn_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rn_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rn_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rn_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rn_s.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers False --remove_special_characters True --remove_stopwords True --stem False

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rsc_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rsc_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rsc_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rsc_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rsc_rsw.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers False --remove_special_characters True --remove_stopwords False --stem True

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rsc_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rsc_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rsc_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rsc_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rsc_s.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers False --remove_special_characters False --remove_stopwords True --stem True

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rsw_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rsw_s.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers True --remove_special_characters False --remove_stopwords False --stem False

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rn.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rn.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rn.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rn.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rn.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers False --remove_special_characters True --remove_stopwords False --stem False

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rsc.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rsc.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rsc.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rsc.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rsc.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers False --remove_special_characters False --remove_stopwords True --stem False

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_rsw.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_rsw.sav



python preprocessor.py --in_path F:/NLP/Project/Introduction_NLP/data/train_data_A.txt --out_path F:/NLP/Project/Introduction_NLP/output/ --remove_numbers False --remove_special_characters False --remove_stopwords False --stem True

python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/Atheism_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/Atheism_stance.tsv --penalty none --solver newton-cg --c 0.1 --filename F:/NLP/Project/Introduction_NLP/output/atheism_LR_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/HillaryClinton_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/HillaryClinton_stance.tsv --penalty none --solver lbfgs --c 0.01 --filename F:/NLP/Project/Introduction_NLP/output/hillary_LR_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/ClimateChangeisaRealConcern_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/climate_LR_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/FeministMovement_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/FeministMovement_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/feminist_LR_s.sav
python models.py --model lg --tfidf_file F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_tfidf.tsv --stance F:/NLP/Project/Introduction_NLP/output/LegalizationofAbortion_stance.tsv --penalty l2 --solver newton-cg --c 1.0 --filename F:/NLP/Project/Introduction_NLP/output/abortion_LR_s.sav



