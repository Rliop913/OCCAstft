

lists = []

bank = []
L = []
R = []

SL = []
SR = []

WR = 3
WS = 1 << WR
for i in range(WS):
    bank.append(i)
for i in bank:
    if i < (WS // 2):
        L.append(i)
        SL.append(0)
    else:
        R.append(i)
        SR.append(0)

def idxing(window_size:int, i_itr:int, segment_size:int, L, R, SL, SR):
    # if segment_size > 1:
    scope = segment_size // 2
    for i in range(i_itr):
        n = i+1
        seg_idx = n % segment_size
        
        if seg_idx < scope:
            SL[i] = L[i]
            print(i, window_size, len(SL), i_itr, i + (window_size // 4) - 2,
             seg_idx, scope, segment_size)
            SL[i + (window_size // 4) - 2] = R[i]
        else:
            SR[i] = L[i]
            print(i, window_size, len(SL), i_itr, i + (window_size // 4) - 2,
             seg_idx, scope, segment_size, "False")
            SR[i + (window_size // 4) - 2] = R[i]
            
    for i in range(i_itr):
        L[i] = SL[i]
        R[i] = SR[i]

    
# i_itr = WS // 2
# segSize = WS // 2
# print(WS)
# for i in range(WR):

#     idxing(WS, i_itr, segSize, L, R, SL, SR)
#     for j in range(i_itr):
        
#         print("L: ", L[j], "R: ", R[j], "IDX: ", j, segSize)
#     segSize /= 2



def optiindexing(Halfsegment:int, Lidx:int, HalfWin):
    bank = [0,0]
    
    print("L: ", bank[0], "R: ", bank[1], "origin: ", Lidx)


def stockhamIndexing(halfwinSize:int, localidx:int, segsize:int):
    temp = [0,0]
    temp[0] = localidx + (localidx & (~(segsize - 1)))
    temp[1] = temp[0] + segsize
    save = [0,0]
    save[0] = temp[0] // (segsize * 2) * segsize
    save[0] = save[0] + temp[0] % segsize
    save[1] = halfwinSize + save[0]
    print("loadidx: ", temp, "saveidx: ", save, "origin: ", localidx)
    pass

def indexing(LOCAL_FIRST_MAXIMUX_ID:int ,powed_stage:int):

    temp = [0,0]
    temp[0] = LOCAL_FIRST_MAXIMUX_ID + (LOCAL_FIRST_MAXIMUX_ID & (~(powed_stage - 1)))
    temp[1] = temp[0] + powed_stage
    return temp

def ttestindexer(segmentSize:int, shift_amount, buff):
    
    # for i in range(100):
    #     buff = i + (i & (~(segmentSize - 1)))
        if ((buff & segmentSize) >> (segmentSize.bit_length() - 1)) ^ 1 == 0:
            buff = buff - segmentSize
            decodeI = (buff & (segmentSize - 1)) + ((buff >> (shift_amount + 1)) << shift_amount)
            print("early returned")
            decodeI += segmentSize
        else:
            decodeI = (buff & (segmentSize - 1)) + ((buff >> (shift_amount + 1)) << shift_amount)
        codeI = decodeI + (decodeI & (~(segmentSize - 1)))
        
        print("IDX: ", codeI, "lload: ", buff, "decodeI: ", decodeI)


def indexingTest(OIdx, stageRadix):
    segmentItr = OIdx >> (stageRadix)
    segmentIndex=OIdx & ((1 << (stageRadix)) -1)
    outIdx= segmentItr * (1 << (stageRadix + 1)) + segmentIndex

    print("out index: ", outIdx)
    print("right pair: ", outIdx + (1 << (stageRadix)))

for i in range(512):
    indexingTest(i + 512, 10, 2)
# 1
# 2
# 4
# 8
# 16
# 32
# 64
# 128
# ttestindexer(64, 6, 66)

# for o_itr in range(0, 8*8, 8):
#     for i_itr in range(0, 8):
#         optiindexing(32, o_itr + i_itr, 64)

# print(indexing(0,2))
# print(indexing(1,2))
# print(indexing(2,2))
# print(indexing(3,2))
# print(indexing(4,2))
# print(indexing(5,2))
# print(indexing(6,2))
# print(indexing(7,2))
# print(indexing(8,2))



def butterfly_match():
    pass

