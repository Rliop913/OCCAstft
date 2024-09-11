#include <catch2/catch_test_macros.hpp>
#include "FFTStruct.hpp"


// TEST_CASE("Test FFT Struct", "[Client Test]")
// {
//     ULL counter = 4132;
//     std::vector<float> testData(10);
//     for(auto& i : testData)
//     {
//         i = 3213;
//     }
//     FFTRequest origin(10, 0.5, counter);
//     origin.MakeSharedMemory(CUDA, testData.size());
//     origin.SetData(testData);

//     auto bintest = origin.Serialize();

//     FFTRequest cloned;
//     cloned.Deserialize(bintest);
//     auto cloneID = cloned.getID();
//     auto cloneOut = cloned.getData();
//     REQUIRE(cloneID == origin.getID());
//     REQUIRE(cloneOut.has_value());
//     for(int i = 0; i < testData.size();++i)
//     {
//         REQUIRE(testData[i] == cloneOut.value()[i]);
//     }
// }