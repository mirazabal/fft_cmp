
Instructions:

cd muFFT && mkdir build && cd build && cmake .. && make -j8
&& 
cd ..
&&
gcc main.c -O3 -march=native muFFT/build/*.a -lm
