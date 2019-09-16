# Data Files  

|   Dir Name   |    Dataset    |  Train |   Dev | Test  |
|:------------:|:-------------:|:------:|------:|-------|
|    english   | CELEX-English |  71502 |  8941 | 8938  |
|    italian   |    Festival   | 327812 | 40978 | 40976 |
|    basque    |     E-Hitz    |  80063 | 10009 | 10007 |
|    NETtalk   |    NETtalk    |  13004 |  3502 | 3501  |
| NETtalkTrain |    NETtalk    |  18006 |  1001 | 1000  |
|     dutch    |  CELEX-Dutch  | 262035 | 32759 | 32754 |
|   manipuri   |  IIT-Guwahati |  13745 |  1718 | 1718  |
|    french    |  OpenLexique  | 110540 | 13818 | 13817 |

### Generating Data Files From Source

1. Open the CELEX sqlite database in 'DB Browser for SQLite'.  
2. Export the desired table to JSON. Ex: optimized_disc.  
3. Move the file to this celex folder.  
4. Generate the `dev.txt`, `train.txt`, and `test.txt` files:  
	`>>> python3 split.py fname.json train_percent test_percent dev_percent`  
5. move generated files to a folder named by language. Ex: english  

If each language directory contains `train.txt`, `val.txt`, and `test.txt`, you should not need to regenerate the data files. These data files were sourced from a privately-managed database.  