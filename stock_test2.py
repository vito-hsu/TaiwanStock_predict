# 本 .py 檔案主要是針對 DL 深度學習模型調教
# 代替 0616.py 檔案 !!!!
import yfinance as yf
from matplotlib import pyplot
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler




def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



k           =   1
stock_name  =   "2330"  
df          =   yf.download(stock_name + ".TW")


while(True):
    #region 變數設定
    # # 8/2 新增 StatefulModel, 故需要 batchsize = 1 的值
    batchsize       =   random.choices([1, 8, 16, 32, 64, 128, 256], k=1, weights=[0.1, 0.01, 0.1, 0.5, 0.1, 0.1, 0.01])[0]   
    data_len        =   600 
    timesteps_head  =   12                                              # 記得改變這邊, 同時也要改下面
    timesteps_tail  =   3                                               # 記得改變這邊, 同時也要改下面
    # n               =   data_len+1-timesteps_head-(timesteps_tail-1)
    var_n           =   2                                               # 總計有兩個變數 var1, var2
    dropout_units   =   random.choices([0.1, 0.2], k=1, weights=[0.5, 0.5])[0] if batchsize !=1 else 0.05
    num_units       =   round(data_len/batchsize)                       # if batchsize != 1 else 100          
    lr              =   0.01                                            # default : 0.001
    mindelta        =   0
    patience        =   25
    stateful        =   1 if batchsize ==1 else 0                       # 很重要
    model_name      =   random.choices(['GRU', 'LSTM'], k=1, weights=[0.1, 0.9])[0] 
    bidirectional   =   random.choices([0, 1], k=1, weights=[0.5, 0.5])[0] 
    l1l2            =   random.choices([0, 1], k=1, weights=[0.5, 0.5])[0]
    twolayer        =   random.choices([0, 1], k=1, weights=[0.5, 0.5])[0] 
    # data_len 這邊是指資料要使用最近的多少筆資料, 注意並非 timesteps_head
    #endregion

    #region 模型建立，共分成 12 種模型，0802 加入 stateful 模型，總計模型數來到 24 = 12*2 個 !!! 
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Bidirectional, LSTM, GRU
    from keras.regularizers import L1L2, L2
    from keras.callbacks import EarlyStopping, LearningRateScheduler
    from keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
    if stateful == 1:
        if l1l2 == 1:
            if bidirectional == 1:
                model           =   Sequential()                                                                                                        # design network
                model.add(Bidirectional(
                    eval(model_name)(
                        units=num_units, 
                        stateful=True,
                        batch_input_shape=(1, timesteps_head, var_n),                                                                                            # input_shape  (1, 8)     train_X.shape[1] = timesteps_head,    train_X.shape[2] = 2
                        kernel_regularizer=L1L2(l1=0.0001, l2=0.001),                                           
                        bias_regularizer=L2(0.001),
                        activity_regularizer=L2(0.0001)
                    )
                ))                    
                model.add(Dropout(dropout_units))
                model.add(Dense(1))  
            else:
                model           =   Sequential()                                                                                                        # design network
                model.add(eval(model_name)(
                    num_units, 
                    stateful=True,
                    batch_input_shape=(1, timesteps_head, var_n),                                                                                                # input_shape  (1, 8)
                    kernel_regularizer=L1L2(l1=0.0001, l2=0.001),
                    bias_regularizer=L2(0.001),
                    activity_regularizer=L2(0.0001)            
                ))                    
                model.add(Dropout(dropout_units))
                model.add(Dense(1))                                                                                                                     # 這裡要跟著 output 數量調整 !!
        else:
            if twolayer == 1:
                if bidirectional == 1:
                    model           =   Sequential()                                                                                                    # design network
                    model.add(Bidirectional(eval(model_name)(num_units, stateful=True, batch_input_shape=(1, timesteps_head, var_n), return_sequences=True)))                    # input_shape  (1, 8)
                    model.add(Dropout(dropout_units))
                    model.add(Bidirectional(eval(model_name)(int(num_units/4))))
                    model.add(Dense(1))  
                else:
                    model           =   Sequential()                                                                                                    # design network
                    model.add(eval(model_name)(num_units, stateful=True, batch_input_shape=(1, timesteps_head, var_n), return_sequences=True))                                  # input_shape  (1, 8)
                    model.add(Dropout(dropout_units))
                    model.add(eval(model_name)(int(num_units/4)))
                    model.add(Dense(1))   
            else:
                if bidirectional == 1:
                    model           =   Sequential()                                                                                                    # design network
                    model.add(Bidirectional(eval(model_name)(num_units, stateful=True, batch_input_shape=(1, timesteps_head, var_n))))                                          # input_shape  (1, 8)
                    model.add(Dropout(dropout_units))
                    model.add(Dense(1))  
                else:
                    model           =   Sequential()                                                                                                    # design network
                    model.add(eval(model_name)(num_units, stateful=True, batch_input_shape=(1, timesteps_head, var_n)))                                                         # input_shape  (1, 8)
                    model.add(Dropout(dropout_units))
                    model.add(Dense(1))      
    else:
        if l1l2 == 1:
            if bidirectional == 1:
                model           =   Sequential()                                                                                                        # design network
                model.add(Bidirectional(
                    eval(model_name)(
                        num_units, 
                        input_shape=(timesteps_head, var_n),                                                                                            # input_shape  (1, 8)     train_X.shape[1] = timesteps_head,    train_X.shape[2] = 2
                        kernel_regularizer=L1L2(l1=0.0001, l2=0.001),                                           
                        bias_regularizer=L2(0.001),
                        activity_regularizer=L2(0.0001)
                    )
                ))                    
                model.add(Dropout(dropout_units))
                model.add(Dense(1))  
            else:
                model           =   Sequential()                                                                                                        # design network
                model.add(eval(model_name)(
                    num_units, 
                    input_shape=(timesteps_head, var_n),                                                                                                # input_shape  (1, 8)
                    kernel_regularizer=L1L2(l1=0.0001, l2=0.001),
                    bias_regularizer=L2(0.001),
                    activity_regularizer=L2(0.0001)            
                ))                    
                model.add(Dropout(dropout_units))
                model.add(Dense(1))                                                                                                                     # 這裡要跟著 output 數量調整 !!
        else:
            if twolayer == 1:
                if bidirectional == 1:
                    model           =   Sequential()                                                                                                    # design network
                    model.add(Bidirectional(eval(model_name)(num_units, input_shape=(timesteps_head, var_n), return_sequences=True)))                    # input_shape  (1, 8)
                    model.add(Dropout(dropout_units))
                    model.add(Bidirectional(eval(model_name)(int(num_units/4))))
                    model.add(Dense(1))  
                else:
                    model           =   Sequential()                                                                                                    # design network
                    model.add(eval(model_name)(num_units, input_shape=(timesteps_head, var_n), return_sequences=True))                                  # input_shape  (1, 8)
                    model.add(Dropout(dropout_units))
                    model.add(eval(model_name)(int(num_units/4)))
                    model.add(Dense(1))   
            else:
                if bidirectional == 1:
                    model           =   Sequential()                                                                                                    # design network
                    model.add(Bidirectional(eval(model_name)(num_units, input_shape=(timesteps_head, var_n))))                                          # input_shape  (1, 8)
                    model.add(Dropout(dropout_units))
                    model.add(Dense(1))  
                else:
                    model           =   Sequential()                                                                                                    # design network
                    model.add(eval(model_name)(num_units, input_shape=(timesteps_head, var_n)))                                                         # input_shape  (1, 8)
                    model.add(Dropout(dropout_units))
                    model.add(Dense(1))      
    #endregion

    #region 模型編譯
    adam    =   Adam(learning_rate = lr)
    RMSp    =   RMSprop(learning_rate = lr)
    Adad    =   Adadelta(learning_rate = lr)
    Adag    =   Adagrad(learning_rate = lr)
    opt     =   random.choices(['adam', 'RMSp', 'Adad', 'Adag'], k=1, weights=[3, 3, 0.0001, 4])[0]       
    model.compile(
        loss        =   'mae', 
        optimizer   =   eval(opt)                                   #,     metrics=['mae']
    )
    # model.summary()
    #endregion

    # 以下決定需要訓練的資料集有哪些 !!
    df1         =   df.iloc[(df.shape[0]-data_len)-1:,:]    if batchsize != 1 else df.iloc[(df.shape[0]-data_len)-1:,:]
    df2         =   df.iloc[(df.shape[0]-data_len*2)-1:,:]  if batchsize != 1 else df.iloc[(df.shape[0]-data_len*2)-1:(df.shape[0]-data_len),:]
    df3         =   df.iloc[(df.shape[0]-data_len*3)-1:,:]  if batchsize != 1 else df.iloc[(df.shape[0]-data_len*3)-1:(df.shape[0]-data_len*2),:]
    df4         =   df.iloc[(df.shape[0]-data_len*4)-1:,:]  if batchsize != 1 else df.iloc[(df.shape[0]-data_len*4)-1:(df.shape[0]-data_len*3),:]


    train_stock =   ["df4", "df3", "df2", "df1"]                                                                                           # ["df5269_1", "df5269_2", "df5269_3", "df5269_4", "df5269_5", "df5269_6"]            
    mae_test = [] ;  mae_Stupid = []

    # 以下開始訓練，並計算訓練所需時間
    import time
    start = time.time()
    for stock in train_stock:
        # stock = "df4"
        dataset     =   eval(stock) 
        linspace_n  =   1                   
        linspace    =   np.linspace(0, data_len, int(data_len/linspace_n)+1) 
        dataset     =   dataset.iloc[linspace].reset_index()
        variable    =   'Close'                                                                                       # random.choices(["Volume", "High", "Low"], k=1, weights=[0.3, 0.3, 0.4])[0] if batchsize != 1 else "Close"                 # 這裡決定要用甚麼輔助變數                                                              # 固定使用 svid_10050
        dataset     =   dataset[['Adj Close', variable]]            

        values      =   dataset.values.astype('float32')                                                                                                        # type(values) : ndarray
        scaler      =   MinMaxScaler(feature_range=(0, 1))                                                                                                      # values.shape (5493, 4)
        scaled      =   scaler.fit_transform(values)                                                                                                            # scaled.shape (5493, 4)
        
        # 以下檢查 務必要跟一開始設定一樣
        if timesteps_head != 12   or timesteps_tail != 3:                            
            break
                                    
        reframed    =   series_to_supervised(scaled, timesteps_head, timesteps_tail)                                # reframed.shape (5485, 36)
        reframed.drop(reframed.columns[[-1, -3, -5]], axis=1, inplace=True)                                         # 注意!!!! 如果timesteps_tail = 3，這裡要改成 reframed.drop(reframed.columns[[-1, -3, -5]], axis=1, inplace=True)
                                                                                                                    # reframed 查看很重要 !!!
        values      =   reframed.values                                                                             # values.shape (5485, 33)


        train_ratio =   random.randrange(35, 36, 1)/100                                                             # 隨機給定訓練量
        valid_ratio =   random.randrange(25, 26, 1)/100                                                             # 隨機給定驗證量
        train_n     =   round(values.shape[0]*train_ratio)                                                                        # 前 train_n 筆數據當作訓練集
        valid_n     =   round(values.shape[0]*valid_ratio)                                                                        # 中間 valid_n 筆數據當作驗證集
        test_n      =   values.shape[0]-train_n-valid_n                                                                           # 最後剩下筆數據當作測試集
        train       =   values[:train_n, :]                                                                         # train.shape   ; type(train) 
        valid       =   values[train_n:train_n+valid_n, :]                                                          # valid.shape 
        test        =   values[train_n+valid_n:, :]                                                                 # test.shape  
        train_X, train_y    =   train[:, :-3], train[:, -3:]                                                        # 注意!!!! 如果timesteps_tail = 3，這裡要改成 train_X, train_y    =   train[:, :-3], train[:, -3:]
        valid_X, valid_y    =   valid[:, :-3], valid[:, -3:]                                                        # train_X.shape
        test_X, test_y      =   test[:, :-3], test[:, -3:]                                                          # 注意!!!! 如果timesteps_tail = 3，這裡要改成   test_X, test_y    =    test[:, :-3], test[:, -3:]      
        train_X     =   train_X.reshape(train_X.shape[0], timesteps_head, int(train_X.shape[1]/timesteps_head))     # reshape input to be 3D [samples, timesteps_head, features]
        valid_X     =   valid_X.reshape(valid_X.shape[0], timesteps_head, int(valid_X.shape[1]/timesteps_head))     # train_X.shape  (177, 5, 2)
        test_X      =    test_X.reshape( test_X.shape[0], timesteps_head, int( test_X.shape[1]/timesteps_head))     # test_X.shape  (77, 5, 2)





        # 模型訓練
        def my_schedule(epoch, patience=patience, lr=lr):                                                                                     # 可以設定讓 lr 越來越快或是越來越慢 !! 目前設定方式是要讓速度越來越快 !!
            if epoch < patience*10:
                return lr
            elif epoch < patience*100:
                return lr*0.5
            elif epoch < patience*1000:
                return lr*0.1
            else:
                return lr

        lr_schedule = LearningRateScheduler(                                                                        # 8/2 新增 學習速率規劃器 !!
            schedule=my_schedule, 
            verbose=1                                                                                               # verbose 0 : quiet   ,    1 : update
        )        

        restore_best_weights = True #if batchsize != 1 else False                                                   # 0804 嘗試修正 !!
        early_stopping  =  EarlyStopping(
            monitor     = 'val_loss',                                                                               # 0804 嘗試修正 !!
            min_delta   = mindelta, 
            patience    = patience,
            restore_best_weights = restore_best_weights                                                             # 建議做時序分析時, 都這樣使用 !!
        )


        history     =   model.fit(                                                                                  # fit network
            train_X,                                                                                                # train_X.shape ; train_y.shape
            train_y,                                                                                                # valid_X.shape ; valid_y.shape
            epochs = 3000,                                                                                          # 0804 嘗試修正 !!
            batch_size = batchsize,                                                                                 # batch_size 對於時序分析來說非常重要 !!!
            validation_data = (valid_X, valid_y),                                                                   # 所有時序分析 model.fit 需要注意的 !!!
            shuffle = False,                                                                                        # 千萬不能亂打亂順序 !!!
            callbacks = [early_stopping, lr_schedule]
        )          


    
    end = time.time()



    # stock == train_stock[-1] 的用意是我們將總資料切分為多等份做訓練
    # 故等到所有資料訓練過，才進行預測 !!!
    # 訓練完後進行預測，並檢視評估模型效能            
    if stateful == 1 and stock == train_stock[-1]:
        model.reset_states()                                                                                    # 可參閱 《Keras深度學習攻略手冊》 4-37
        yhat        =   model.predict_generator(test_X, steps=test_X.shape[0])
        yhat        =   yhat.reshape(yhat.shape[0], 1)                                                          # yhat.shape (150, 1)
        test_X      =   test_X.reshape((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))                      # test_X.shape from (150, 8, 4) to  (150, 32)
        inv_yhat    =   np.concatenate((yhat, test_X[:, test_X.shape[1]-1:]), axis=1)                           # inv_yhat.shape (150, 4)     ;  注意： 有幾個變數就減幾  (test_X.shape[1]後面)                       
        inv_yhat    =   scaler.inverse_transform(inv_yhat)                                                      # inv_yhat.shape (150, 4)
        inv_yhat    =   inv_yhat[:,0]         
        test_y      =   test_y[:,-1].reshape(test_y.shape[0], 1)                                                # 0623更新!! 延長預測的關鍵在這 (test_y) !!  test_y.shape (1360, 1)
        inv_y       =   np.concatenate((test_y, test_X[:, test_X.shape[1]-1:]), axis=1)                         # inv_y.shape (1360, 4)     ;  注意： 有幾個變數就減幾    
        inv_y       =   scaler.inverse_transform(inv_y)                                                         # inv_y.shape (1360, 4)
        inv_y       =   inv_y[:,0]
    elif stock == train_stock[-1]:
        yhat        =   model.predict(test_X)[:,-1]                                                             # 0623更新!! 延長預測的關鍵不在這 (test_X)!!  yhat.shape (150, )
        yhat        =   yhat.reshape(yhat.shape[0], 1)                                                          # yhat.shape (150, 1)
        test_X      =   test_X.reshape((test_X.shape[0], test_X.shape[1]*test_X.shape[2]))                      # test_X.shape from (150, 8, 4) to  (150, 32)
        inv_yhat    =   np.concatenate((yhat, test_X[:, test_X.shape[1]-1:]), axis=1)                           # inv_yhat.shape (150, 4)     ;  注意： 有幾個變數就減幾  (test_X.shape[1]後面)                       
        inv_yhat    =   scaler.inverse_transform(inv_yhat)                                                      # inv_yhat.shape (150, 4)
        inv_yhat    =   inv_yhat[:,0]                                            
        # invert scaling for actual                                                                     
        test_y      =   test_y[:,-1].reshape(test_y.shape[0], 1)                                                # 0623更新!! 延長預測的關鍵在這 (test_y) !!  test_y.shape (1360, 1)
        inv_y       =   np.concatenate((test_y, test_X[:, test_X.shape[1]-1:]), axis=1)                         # inv_y.shape (1360, 4)     ;  注意： 有幾個變數就減幾    
        inv_y       =   scaler.inverse_transform(inv_y)                                                         # inv_y.shape (1360, 4)
        inv_y       =   inv_y[:,0]


    from sklearn.metrics import mean_absolute_error
    mae_test.append(mean_absolute_error(inv_y[3:], inv_yhat[3:]))                                               # calculate MAE of test data with GRU method
    mae_Stupid.append(mean_absolute_error(inv_y[3:], inv_y[:-3]))                                               # calculate MAE of test data with stupid method
    #endregion`

    from os.path import isfile
    from datetime import datetime, date
    today = date.today()
    now = datetime.now()
    save_date = "1111"    

    # log 存放位置 !!!!!!!!!!!!!!!
    if isfile(rf"model/stock_{save_date}.txt"):
        a = open(rf"model/stock_{save_date}.txt", "a")
        wsting = f"model={model_name}, bi={bidirectional}, l1l2={l1l2}, two={twolayer}, state={stateful}, d.len={data_len}, t.df={stock_name}, lin={linspace_n}, t.head={timesteps_head}, d.o.={dropout_units}, units={num_units}, lr={lr}, opt={opt}, b.s.={batchsize}, pa.={patience}, var={variable}, mae_t-S={mae_test[-1]-mae_Stupid[-1]:.1f}, r.b.w={restore_best_weights}, r.t.={round(end-start)}"  # loss-val_loss={min(history.history['loss'])-min(history.history['val_loss']):.3f}
        a.writelines(wsting+"\n")
        a.close()
    else:
        b = open(rf"model/stock_{save_date}.txt","w")
        wsting = f"model={model_name}, bi={bidirectional}, l1l2={l1l2}, two={twolayer}, state={stateful}, d.len={data_len}, t.df={stock_name}, lin={linspace_n}, t.head={timesteps_head}, d.o.={dropout_units}, units={num_units}, lr={lr}, opt={opt}, b.s.={batchsize}, pa.={patience}, var={variable}, mae_t-S={mae_test[-1]-mae_Stupid[-1]:.1f}, r.b.w={restore_best_weights},  r.t.={round(end-start)}"  # loss-val_loss={min(history.history['loss'])-min(history.history['val_loss']):.3f}
        b.writelines(wsting+"\n")
        b.close()

    #region part3
    # loss, val_loss plot
    pyplot.plot(history.history['loss'], label='train_loss')                                           # plot history
    pyplot.plot(history.history['val_loss'], label='valid_loss')
    pyplot.legend()
    # if mae_test[-1]-mae_Stupid[-1] < -1 :
    #   pyplot.show()    
    pyplot.savefig(rf"model/stock_{save_date}/"+f'{k}.pic1_' + today.strftime("%m-%d-") + now.strftime("%H-%M-%S") + f"-mae_test_diff-{mae_test[-1]-mae_Stupid[-1]:.3f}.jpg", dpi=400)
    pyplot.close()

    pyplot.plot(inv_y[3:], label = 'Real')
    pyplot.plot(inv_y[:-3], label = 'Stupid predicted')
    pyplot.plot(inv_yhat[3:], label = 'LSTM predicted')
    pyplot.title(f"val_loss={np.min(history.history['val_loss']):.3f}, state={stateful}, d.o.={dropout_units}, mae.diff={mae_test[-1]-mae_Stupid[-1]:.3f}")                                 # 這邊跟 early_stopping 裡面的 restore_best_weights 有關 !!
    pyplot.legend()
    # if mae_test[-1]-mae_Stupid[-1] < -1 :
    #   pyplot.show()  
    pyplot.savefig(rf"model/stock_{save_date}/"+f'{k}.pic2_' + today.strftime("%m-%d-") + now.strftime("%H-%M-%S") + f"-mae_test_diff-{mae_test[-1]-mae_Stupid[-1]:.3f}.jpg", dpi=400)
    pyplot.close()

    # 儲存模型 save model
    model.save(rf"model/stock_{save_date}/"+f'{k}.multi_model_' + today.strftime("%m_%d") + f"_{variable}_{linspace_n}_jj_3.h5")      
    #endregion


    k += 1







