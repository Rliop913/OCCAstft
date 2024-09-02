@0xa94629bf0af1f2a4;
struct RequestCapnp {
    sharedMemory @0 :Text;
    data @1 :List(Float32);
    mappedID @2 :Text;

    memPTR @3 :UInt64;
    posixFileDes @4 :Int64;
    windowsHandlePTR @5 :UInt64;

    windowRadix @6 :UInt32;
    overlapRatio @7 :Float32;
    dataLength @8 :UInt64;
    overlapdataLength @9 :UInt64;
}