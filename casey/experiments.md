Experiments
===========

-	Fake data (given full R; no bias) --- reconstruction.py
	Results:

       N	       D	       K	   alpha	    beta	     eps	   steps	    RMSE
      10	      10	       1	0.001000	0.020000	0.000050	    1083	0.827328
      10	      10	       1	0.001000	0.020000	0.000050	    1110	1.308441
      10	      10	       1	0.001000	0.020000	0.000050	    1726	0.857964
      10	      10	       1	0.001000	0.020000	0.000050	    1412	1.175466
      10	      10	       1	0.001000	0.020000	0.000050	    1166	1.351235
      10	      10	       1	0.001000	0.020000	0.000050	    1571	1.012256
      10	      10	       1	0.001000	0.020000	0.000005	    1249	0.991950
      10	      10	       1	0.001000	0.020000	0.000005	    1973	1.026269

      10	      10	       2	0.001000	0.020000	0.000050	    1999	0.382506
      10	      10	       2	0.001000	0.020000	0.000050	    1210	0.561100
      10	      10	       2	0.001000	0.020000	0.000050	    1999	0.939934
      10	      10	       2	0.001000	0.020000	0.000050	    1792	0.458905
      10	      10	       2	0.001000	0.020000	0.000050	    1999	0.775775

       N	       D	       K	   alpha	    beta	     eps	   steps	    RMSE
     100	     100	       2	0.001000	0.020000	0.000050	     282	7.390202
     100	     100	       3	0.001000	0.020000	0.000050	     659	8.204818

-	Fake data (given full R; bias) --- reconstruction.py
	
       N	       D	       K	   alpha	    beta	     eps	   steps	    RMSE
     100	     100	       3	0.001000	0.020000	0.000050	     579	27.676833
     100	     100	       3	0.001000	0.020000	0.000050	     734	29.181141

-	Fake data (partial data set; withhold 10%) --- reconstruction.py

-	Cross-validation (small data set, no biases)


-	Cross-validation (small data set, with biases)
	
       N	       D	       K	   alpha	    beta	     eps	   steps	  points	  w/held	 discard	    RMSE
   12787	  131378	       1	0.001000	0.020000	0.000050	     199	  200000	    2000	  180000	0.890601
   12787	  131378	       1	0.001000	0.020000	0.000050	     199	  200000	    2000	  180000	0.867243
   12787	  131378	       1	0.001000	0.020000	0.000050	     499	  200000	    2000	  180000	0.908410
   12787	  131378	       2	0.001000	0.020000	0.000050	     199	  200000	    2000	  180000	0.933478

-	Cross-validation (full data set, no biases)

-	Cross-validation (full data set---witholding 2000, with biases)

       N	       D	       K	   alpha	    beta	     eps	   steps	  points	  w/held	 discard	    RMSE
   12787	  131378	       5	0.000500	0.020000	0.000050	     499	  200000	    2000	       0	0.939141
