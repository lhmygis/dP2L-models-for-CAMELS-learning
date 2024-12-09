import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from keras.models import Model
from keras.layers import Input, Concatenate
from keras import optimizers, callbacks
from datetime import datetime, timedelta

import keras.models
from keras.utils.generic_utils import get_custom_objects
from keras import initializers, constraints, regularizers
from keras.layers import Layer, Dense, Lambda, Activation
import keras.backend as K
import tensorflow as tf

## Import libraries developed by this study
from dP2L_class import regional_dP2L,LSTMq
from dataprocess import DataforIndividual
import loss

## Ignore all the warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = '0'

working_path = 'G:\\hydro_dl_project'   #The root directory where the model project files are located (including dataprocess.py, dP2L_class.py, loss.py)
attrs_path = 'G:\\hydro_dl_project\\camels_data\\datafor_rm\\531_27attrs_scal.csv'   #csv file of standardized values of 27 watershed attributes for 531 CAMELS watersheds

training_start = '1990-10-01'
training_end = '2010-09-30'


#CAMELS watershed IDs
basin_id = [
'1022500',
'1031500',
'1047000',
'1052500',
'1054200',
'1055000',
'1057000',
'1073000',
'1078000',
'1123000',
'1134500',
'1137500',
'1139000',
'1139800',
'1142500',
'1144000',
'1162500',
'1169000',
'1170100',
'1181000',
'1187300',
'1195100',
'4296000',
'1333000',
'1350000',
'1350080',
'1350140',
'1365000',
'1411300',
'1413500',
'1414500',
'1415000',
'1423000',
'1434025',
'1435000',
'1439500',
'1440000',
'1440400',
'1451800',
'1466500',
'1484100',
'1487000',
'1491000',
'1510000',
'1516500',
'1518862',
'1532000',
'1539000',
'1542810',
'1543000',
'1543500',
'1544500',
'1545600',
'1547700',
'1548500',
'1549500',
'1550000',
'1552000',
'1552500',
'1557500',
'1567500',
'1568000',
'1580000',
'1583500',
'1586610',
'1591400',
'1594950',
'1596500',
'1605500',
'1606500',
'1632000',
'1632900',
'1634500',
'1638480',
'1639500',
'1644000',
'1664000',
'1666500',
'1667500',
'1669000',
'1669520',
'2011400',
'2013000',
'2014000',
'2015700',
'2016000',
'2017500',
'2018000',
'2027000',
'2027500',
'2028500',
'2038850',
'2046000',
'2051500',
'2053200',
'2053800',
'2055100',
'2056900',
'2059500',
'2064000',
'2065500',
'2069700',
'2070000',
'2074500',
'2077200',
'2081500',
'2082950',
'2092500',
'2096846',
'2102908',
'2108000',
'2111180',
'2111500',
'2112120',
'2112360',
'2118500',
'2125000',
'2128000',
'2137727',
'2140991',
'2143000',
'2143040',
'2149000',
'2152100',
'2177000',
'2178400',
'2193340',
'2196000',
'2198100',
'2202600',
'2212600',
'2215100',
'2216180',
'2221525',
'2231000',
'2245500',
'2246000',
'2296500',
'2297155',
'2297310',
'2298123',
'2298608',
'2299950',
'2300700',
'2342933',
'2349900',
'2350900',
'2361000',
'2363000',
'2369800',
'2371500',
'2372250',
'2374500',
'2381600',
'2384540',
'2395120',
'2415000',
'2427250',
'2430085',
'2450250',
'2464000',
'2464360',
'2465493',
'2469800',
'2472000',
'2472500',
'2479155',
'2479300',
'2479560',
'2481000',
'2481510',
'4015330',
'4024430',
'4027000',
'4040500',
'4043050',
'4045500',
'4057510',
'4057800',
'4059500',
'4063700',
'4074950',
'4105700',
'4115265',
'4122200',
'4122500',
'4127918',
'4127997',
'4161580',
'4185000',
'4196800',
'4197100',
'4197170',
'4213000',
'4213075',
'4216418',
'4221000',
'4224775',
'4233000',
'4256000',
'3010655',
'3011800',
'3015500',
'3021350',
'3026500',
'3028000',
'3049000',
'3049800',
'3050000',
'3069500',
'3070500',
'3076600',
'3078000',
'3140000',
'3144000',
'3170000',
'3173000',
'3180500',
'3182500',
'3186500',
'3237280',
'3237500',
'3238500',
'3241500',
'3280700',
'3281500',
'3285000',
'3291780',
'3338780',
'3340800',
'3346000',
'3364500',
'3366500',
'3368000',
'3384450',
'3439000',
'3455500',
'3456500',
'3460000',
'3463300',
'3471500',
'3473000',
'3479000',
'3488000',
'3498500',
'3500000',
'3500240',
'3504000',
'3574500',
'3592718',
'3604000',
'5291000',
'5362000',
'5393500',
'5399500',
'5408000',
'5413500',
'5414000',
'5444000',
'5454000',
'5458000',
'5466500',
'5487980',
'5488200',
'5489000',
'5495000',
'5495500',
'5501000',
'5503800',
'5507600',
'5508805',
'5525500',
'5556500',
'5584500',
'5591550',
'5592050',
'5592575',
'5593575',
'5593900',
'5595730',
'7291000',
'7359610',
'7362100',
'7362587',
'7375000',
'8013000',
'8014500',
'5057200',
'5120500',
'6221400',
'6224000',
'6278300',
'6280300',
'6289000',
'6291500',
'6311000',
'6332515',
'6339100',
'6344600',
'6350000',
'6352000',
'6404000',
'6406000',
'6409000',
'6431500',
'6440200',
'6447500',
'6470800',
'6477500',
'6479215',
'6601000',
'6614800',
'6622700',
'6623800',
'6632400',
'6746095',
'6803510',
'6803530',
'6814000',
'6847900',
'6853800',
'6876700',
'6878000',
'6879650',
'6885500',
'6888500',
'6889200',
'6889500',
'6892000',
'6903400',
'6906800',
'6910800',
'6911900',
'6917000',
'6918460',
'6919500',
'6921070',
'6921200',
'7057500',
'7060710',
'7066000',
'7083000',
'7142300',
'7145700',
'7167500',
'7180500',
'7184000',
'7195800',
'7196900',
'7197000',
'7208500',
'7261000',
'7263295',
'7299670',
'7301410',
'7315200',
'7315700',
'7335700',
'7340300',
'7346045',
'8023080',
'8050800',
'8066200',
'8066300',
'8070000',
'8070200',
'8082700',
'8086212',
'8086290',
'8101000',
'8103900',
'8104900',
'8109700',
'8150800',
'8158700',
'8158810',
'8164300',
'8164600',
'8165300',
'8171300',
'8175000',
'8176900',
'8178880',
'8189500',
'8190000',
'8190500',
'8194200',
'8195000',
'8196000',
'8198500',
'8200000',
'8202700',
'8267500',
'8269000',
'8271000',
'8324000',
'8377900',
'8378500',
'8380500',
'9035800',
'9035900',
'9047700',
'9065500',
'9066000',
'9066200',
'9066300',
'9081600',
'9107000',
'9210500',
'9223000',
'9306242',
'9312600',
'9352900',
'9378170',
'9386900',
'9404450',
'9430600',
'9447800',
'9484600',
'9492400',
'9494000',
'9497980',
'9505350',
'9505800',
'9508300',
'9510200',
'9512280',
'9513780',
'10234500',
'10244950',
'10336645',
'10336660',
'10343500',
'12010000',
'12013500',
'12020000',
'12025700',
'12035000',
'12040500',
'12041200',
'12048000',
'12054000',
'12056500',
'12073500',
'12082500',
'12092000',
'12114500',
'12115000',
'12117000',
'12143600',
'12144000',
'12145500',
'12147500',
'12167000',
'12175500',
'12178100',
'12186000',
'12189500',
'12375900',
'12377150',
'12381400',
'12383500',
'12390700',
'12411000',
'12447390',
'12451000',
'12488500',
'13011500',
'13011900',
'13018300',
'13023000',
'13161500',
'13235000',
'13240000',
'13313000',
'13331500',
'14020000',
'14096850',
'14137000',
'14138800',
'14138870',
'14138900',
'14139800',
'14141500',
'14154500',
'14158790',
'14166500',
'14182500',
'14185000',
'14185900',
'14187000',
'14216500',
'14222500',
'14236200',
'14301000',
'14303200',
'14305500',
'14306340',
'14306500',
'14308990',
'14309500',
'14316700',
'14325000',
'14362250',
'14400000',
'10259000',
'11124500',
'11141280',
'11143000',
'11148900',
'11151300',
'11176400',
'11230500',
'11237500',
'11264500',
'11266500',
'11284400',
'11381500',
'11451100',
'11468500',
'11473900',
'11475560',
'11476600',
'11478500',
'11480390',
'11481200',
'11482500',
'11522500',
'11523200',
'11528700',
'11532500'
]
print(len(basin_id))

all_list = []
all_list1 = []

for i in range(len(basin_id)):
    a = basin_id[i]

    if len(basin_id[i]) == 7:
        basin_id[i] = '0' + basin_id[i]

    hydrodata = DataforIndividual(working_path, basin_id[i]).load_data()

    train_set = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]
    train_set1 = hydrodata[hydrodata.index.isin(pd.date_range(training_start, training_end))]

    if a.startswith('0'):
        single_basin_id = a[1:]

    else:
        single_basin_id = a



    static_x = pd.read_csv(attrs_path)
    static_x = static_x.set_index('gauge_id')
    rows_bool = (static_x.index == int(single_basin_id))
    rows_list = [i for i, x in enumerate(rows_bool) if x]
    rows_int = int(rows_list[0])
    static_x_np = np.array(static_x)


    local_static_x = static_x_np[rows_int, :]  # basin_id index in attrs_path
    local_static_x_for_test = np.expand_dims(local_static_x, axis=0)


    local_static_x_for_train = np.expand_dims(local_static_x, axis=0)
    local_static_x_for_train = local_static_x_for_train.repeat(train_set.shape[0], axis=0)



    result = np.concatenate((train_set, local_static_x_for_train), axis=-1)
    result1 = np.concatenate((train_set1, local_static_x_for_train), axis=-1)



    all_list.append(result)
    all_list1.append(result1)

result_ = all_list[0]
result1_ = all_list1[0]


for i in range(len(all_list)-1):
    result_ = np.concatenate((result_, all_list[i+1]), axis=0)
    result1_ = np.concatenate((result1_, all_list1[i+1]), axis=0)

print(result_.shape)
print(result1_.shape)

sum_result = result_[:,
             [0, 1, 2, 3, 4, 32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 5]]

sum_result1 = result1_[:,
             [0, 1, 2, 3, 4, 32, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
              29, 30, 31, 5]]


def generate_train_test(train_set, train_set1, wrap_length):
    train_set_ = pd.DataFrame(train_set)
    train_x_np = train_set_.values[:, :-1]


    print("prcp_mean:", np.mean(train_x_np[:, 0:1]))
    print("tmean_mean:", np.mean(train_x_np[:, 1:2]))
    print("dayl_mean:", np.mean(train_x_np[:, 2:3]))
    print("srad_mean:", np.mean(train_x_np[:, 3:4]))
    print("vp_mean:", np.mean(train_x_np[:, 4:5]))

    print("prcp_std:", np.std(train_x_np[:, 0:1]))
    print("tmean_std:", np.std(train_x_np[:, 1:2]))
    print("dayl_std:", np.std(train_x_np[:, 2:3]))
    print("srad_std:", np.std(train_x_np[:, 3:4]))
    print("vp_std:", np.std(train_x_np[:, 4:5]))


    train_set1_ = pd.DataFrame(train_set1)
    train_x_np1 = train_set1_.values[:, :-1]
    train_x_np1[:,0:1] = (train_x_np1[:,0:1] - 3.412180875008701)/8.063616135480709
    train_x_np1[:,1:2] = (train_x_np1[:,1:2] - 10.50360728383252)/10.30964231561827
    train_x_np1[:,2:3] = (train_x_np1[:,2:3] - 0.49992111027762387)/0.08233807739244361
    train_x_np1[:,3:4] = (train_x_np1[:,3:4] - 339.0181683060079)/131.70378837635886
    train_x_np1[:,4:5] = (train_x_np1[:,4:5] - 975.706859343215)/658.7769190674522


    train_y_np = train_set_.values[:, -1:]


    print("Q_mean:",  np.mean(train_y_np[:,-1:]))
    print("Q_std:",  np.std(train_y_np[:,-1:]))


    wrap_number_train = (train_x_np.shape[0] - wrap_length) // 7 + 1

    train_x = np.empty(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_x1 = np.empty(shape=(wrap_number_train, wrap_length, train_x_np1.shape[1]))
    train_y = np.empty(shape=(wrap_number_train, wrap_length, train_y_np.shape[1]))


    for i in range(wrap_number_train):
        train_x[i, :, :] = train_x_np[i * 7:(wrap_length + i * 7), :]
        train_x1[i, :, :] = train_x_np1[i * 7:(wrap_length + i * 7), :]
        train_y[i, :, :] = train_y_np[i * 7:(wrap_length + i * 7), :]

    return train_x, train_x1, train_y


wrap_length = 270  # It can be other values, but recommend this value should not be less than 180 days
train_x, train_x1, train_y = generate_train_test(sum_result, sum_result1, wrap_length=wrap_length)

print(f'The shape of train_x, train_x1, train_y after wrapping by {wrap_length} days are:')
print(f'{train_x.shape}, {train_x1.shape}, {train_y.shape}')



def create_model(input_xd_shape, input_xd_shape1, hodes, seed):
    xd_input_for_dP2L = Input(shape=input_xd_shape, batch_size=321, name='Input_xd1')
    xd_input_for_lstmq = Input(shape=input_xd_shape1, batch_size=321, name='Input_xd2')


    hydro_output = regional_dP2L(mode='normal', h_nodes = hodes, seed = seed, name='Regional_dP2L_PUB')(xd_input_for_dP2L)

    xd_hydro = Concatenate(axis=-1, name='Concat')([xd_input_for_lstmq[5:], hydro_output])

    lstm_hn = LSTMq(input_xd = 30, hidden_size=256, seed=seed, name='LSTMq')(xd_hydro)

    fc_out = Dense(units=1)(lstm_hn)

    #fc_out = K.permute_dimensions(fc_out, pattern=(1, 0, 2))  # for test model

    model = Model(inputs=[xd_input_for_dP2L, xd_input_for_lstmq[5:]], outputs=fc_out)
    return model


def train_model(model, train_x1, train_x2, train_y, ep_number, lrate, save_path):
    save = callbacks.ModelCheckpoint(save_path, verbose=0, save_best_only=True, monitor='nse_metrics', mode='max',
                                     save_weights_only=True)

    es = callbacks.EarlyStopping(monitor='nse_metrics', mode='max', verbose=1, patience=20, min_delta=0.005,
                                 restore_best_weights=True)

    reduce = callbacks.ReduceLROnPlateau(monitor='nse_metrics', factor=0.8, patience=5, verbose=1, mode='max',
                                         min_delta=0.005, cooldown=0, min_lr=lrate / 100)

    tnan = callbacks.TerminateOnNaN()

    model.compile(loss= loss.nse_loss, metrics=[loss.nse_metrics],
                  optimizer=tf.keras.optimizers.Adam(learning_rate=lrate,clipnorm=1.0))

    history = model.fit(x=[train_x1, train_x2], y=train_y, epochs=ep_number, batch_size=321,
                        callbacks=[save, es, reduce, tnan])
    return history




save_path_models = f'G:/hydro_dl_project/dPL_Experiment/Models_h5/JOH/dP2L1.h5'

model = create_model(input_xd_shape=(train_x.shape[1], train_x.shape[2]), input_xd_shape1=(train_x1.shape[1], train_x1.shape[2]),
                     hodes = 64, seed = 101)
model.summary()

prnn_ealstm_history = train_model(model=model, train_x1=train_x,train_x2=train_x1[5:],
                                  train_y=train_y, ep_number=100, lrate=0.001, save_path=save_path_models)
