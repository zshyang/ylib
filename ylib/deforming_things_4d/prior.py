'''
author
    zhangsihao yang

logs
    2023-10-03
        file created
'''
import json

# 59 animals
animal_names = [
    "bear3EP",
    "bear84Q",
    "bear9AK",
    "bearPDD",
    "bearVGG",
    "bucksYJL",
    "bullMJ6",
    "bunnyQ",           # not in SMAL
    "canieLTT",
    "catBG",
    "cattleAFK",
    "cetaceaUW",
    "chickenDC",        # not in SMAL
    "crocodileOPW",     # not in SMAL
    "deer2MB",
    "deerA4K",
    "deerFEL",
    "deerK5L",
    "deerLO1",
    "deerOMG",
    "deerSPL",
    "deerVMW",
    "dinoBDK",          # not in SMAL
    "dinoET",           # not in SMAL
    "doggieMN5",
    "dragonOF2",        # not in SMAL
    "dragonOLO",        # not in SMAL
    "dragonQKS",        # not in SMAL
    "duck",             # not in SMAL
    "elephantLN0",      # not in SMAL
    "elkML",
    "foxWDFS",
    "foxXAT",
    "foxYSY",
    "foxZED",
    "goatS4J6Y",
    "grizzRSS",
    "hippoDG",
    "hogRD",
    "huskydog3T",
    "leopardSLM",
    "lionessHTR42",
    "milkcow10L",
    "moose1DOG",
    "moose45D",
    "moose6OK9",
    "pigSTK69",
    "procySTEM",        # not in SMAL
    "pumaRW",
    "rabbit7L6",        # not in SMAL
    "rabbitFKP",        # not in SMAL
    "rabbitJM6",        # not in SMAL
    "raccoonVGG",       # not in SMAL
    "ravenOL",          # not in SMAL
    "rhinoDJ7S",        # not in SMAL
    "seabirdKK",        # not in SMAL
    "sheepYZR10",
    "tigerD8H",
    "zebraBM",
]

# animal names that could be coverd by SMAL. 40 animals
smal_animal_names = [
    "bear3EP",
    "bear84Q",
    "bear9AK",
    "bearPDD",
    "bearVGG",
    "bucksYJL",
    "bullMJ6",
    "canieLTT",
    "catBG",
    "cattleAFK",
    "deer2MB",
    "deerA4K",
    "deerFEL",      # 2nd round not in SMAL
    "deerK5L",
    "deerLO1",
    "deerOMG",      # 2nd round not in SMAL
    "deerSPL",
    "deerVMW",
    "doggieMN5",
    "elkML",
    "foxWDFS",
    "foxXAT",
    "foxYSY",
    "foxZED",
    "goatS4J6Y",    # 2nd round not in SMAL
    "grizzRSS",
    "hippoDG",
    "hogRD",
    "huskydog3T",
    "leopardSLM",
    "lionessHTR42",
    "milkcow10L",
    "moose1DOG",
    "moose45D",
    "moose6OK9",    # 2nd round not in SMAL
    "pigSTK69",
    "pumaRW",
    "sheepYZR10",
    "tigerD8H",
    "zebraBM",
]

with open('./deforming_things_4d_stats.json', 'r', encoding='utf-8') as f:
    stats = json.load(f)

_MAPPING_STATEMENT = '''
the mapping between the anime mesh and the Animal3D image path.
the requirement for anime mesh is that the mesh should be close to resting
pose.
the requirement for Animal3D image is that the image should be appeared as 
close to the anime mesh as possible. 
both of these two files are picked by visual inspection.
'''

file_mapping = {
    # chatgpt thinks it's a brown bear cause:
    # it appears to have a more prominent shoulder hump, a flatter face
    # profile, and a larger, more robust body shape.
    # 更为突出的肩部驼峰，更为平坦的脸部轮廓，以及更大、更健壮的身体形态
    'bear3EP_Agression/bear3EP_Agression.anime':
        'n02132136/n02132136_14294.JPEG',
    # brown bear
    'bear84Q_Agression/bear84Q_Agression.anime':
        'n02132136/000000235064_brown_bear.jpg',
    # brown bear
    'bear9AK_Agression/bear9AK_Agression.anime':
        'n02132136/n02132136_10767.JPEG',
    # 野猪
    'bearPDD_Idle1/bearPDD_Idle1.anime':
        'n02397096/n02397096_2206.JPEG',
    # brown bear
    'bearVGG_Idle1/bearVGG_Idle1.anime':
        'n02132136/000000235064_brown_bear.jpg',
    # 虽然是叫雄鹿，可是看起来既像鹿又像马。对应的照片，虽然有角，可是实际的mesh是
    # 没有角的。所以这里用的是大角羊（n02415577）的照片。
    'bucksYJL_Idle3/bucksYJL_Idle3.anime':
        '/n02415577/n02415577_2587.JPEG',
    # 牛
    'bullMJ6_Death2/bullMJ6_Death2.anime':
        'n02403003/000000041700_cow.jpg',
    # 犬科动物，感觉上是狐狸
    'canieLTT_Idles0/canieLTT_Idles0.anime':
        'n02120079/n02120079_32157.JPEG',
    # 猫，这个就一个动作，而且看不到脸。选的是豹子的照片
    'catBG_run/catBG_run.anime':
        'n02128385/n02128385_24490.JPEG',
    # 牛
    'cattleAFK_Idle3/cattleAFK_Idle3.anime':
        'n02403003/000000041700_cow.jpg',
    # （鹿）
    'deer2MB_Idle3/deer2MB_Idle3.anime':
        '/n02415577/n02415577_2587.JPEG',
    # （鹿）
    'deerA4K_Idle1/deerA4K_Idle1.anime':
        '/n02415577/n02415577_2587.JPEG',
    # （鹿）
    'deerK5L_eat1/deerK5L_eat1.anime':
        '/n02415577/n02415577_2587.JPEG',
    # （鹿）
    'deerLO1_death1/deerLO1_death1.anime':
        '/n02415577/n02415577_2587.JPEG',
    # （鹿）
    'deerSPL_Actions0/deerSPL_Actions0.anime':
        '/n02415577/n02415577_2587.JPEG',
    # （鹿）
    'deerVMW_Idle1/deerVMW_Idle1.anime':
        '/n02415577/n02415577_2587.JPEG',
    # 小狗可以和小狐狸类似
    'doggieMN5_idle1/doggieMN5_idle1.anime':
        'n02096177/n02096177_7217.JPEG',
    # （麋鹿）有点驼背的鹿，不知道概算牛还是马，先暂时用一张牛的来弄。
    'elkML_Actions0/elkML_Actions0.anime':
        'n02403003/000000117131_cow.jpg',
    # （狐狸）
    'foxWDFS_Actions0/foxWDFS_Actions0.anime':
        'n02120079/n02120079_32157.JPEG',
    # 这个是一只小狐狸类似一个小狗（狐狸），一会儿在狗里面找一下。
    'foxXAT_Attack2/foxXAT_Attack2.anime':
        'n02109961/n02109961_8837.JPEG',
    # （狐狸）
    'foxYSY_Agression/foxYSY_Agression.anime':
        'n02120079/n02120079_32157.JPEG',
    # 这个是一只小狐狸类似一个小狗（狐狸），一会儿在狗里面找一下。
    'foxZED_idle1/foxZED_idle1.anime':
        'n02109961/n02109961_8837.JPEG',
    # 灰熊
    'grizzRSS_Actions1/grizzRSS_Actions1.anime':
        'n02132136/000000235064_brown_bear.jpg',
    # （河马）这个可能就直接从河马的初始开始寻找了，没有对应的照片
    'hippoDG_bite/hippoDG_bite.anime':
        'n02397096/n02397096_1156.JPEG',
    # 野猪，这个猪比较胖，有一个细细的尾巴，然后没有獠牙。以后在猪这一类里面找一下
    'hogRD_Actions0/hogRD_Actions0.anime':
        'n02397096/n02397096_1156.JPEG',
    # （哈士奇）
    'huskydog3T_Actions3/huskydog3T_Actions3.anime':
        'n02109961/n02109961_19823.JPEG',
    # （豹）
    'leopardSLM_roam/leopardSLM_roam.anime':
        'n02128385/n02128385_24490.JPEG',
    # （母狮）
    'lionessHTR42_action2/lionessHTR42_action2.anime':
        'n02125311/n02125311_51356.JPEG',
    # 奶牛
    'milkcow10L_Attack1/milkcow10L_Attack1.anime':
        'n02403003/000000041700_cow.jpg',
    # （麋鹿）
    'moose1DOG_Attack1/moose1DOG_Attack1.anime':
        '/n02415577/n02415577_2587.JPEG',
    # （麋鹿）
    'moose45D_Attack1/moose45D_Attack1.anime':
        '/n02415577/n02415577_2587.JPEG',
    # （猪）非常肥硕的一只猪就是说。
    'pigSTK69_eat1/pigSTK69_eat1.anime':
        'n02397096/n02397096_1156.JPEG',
    # 美洲狮
    'pumaRW_Idles1/pumaRW_Idles1.anime':
        'n02125311/n02125311_51356.JPEG',
    # （羊）小羊，
    'sheepYZR10_getHit1/sheepYZR10_getHit1.anime':
        'n02109961/n02109961_8837.JPEG',
    # （虎）
    'tigerD8H_Idles0/tigerD8H_Idles0.anime':
        'n02129604/n02129604_25069.JPEG',
    # （斑马）
    'zebraBM_eat/zebraBM_eat.anime':
        'n02391049/000000001777_zebra.jpg',
}
