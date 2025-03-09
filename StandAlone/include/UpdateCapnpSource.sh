#!/bin/bash
capnp compile -oc++ FFTcapnp.capnp
cp FFTcapnp.capnp.c++ ../src/
rm FFTcapnp.capnp.c++
