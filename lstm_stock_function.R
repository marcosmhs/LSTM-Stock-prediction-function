lstm_stock <- function(
  origin = "^GSPC",
  stock_name,
  start_date,
  end_date = Sys.Date(),
  lstm_executions = 20,
  arima_compare = FALSE,
  training_data_percentage = 0.8) {

  # libraries block, if some library is not present it will be installed.
  pacotes <- c("ggplot2", "plotly", "dplyr", "forecast", "reshape2", "keras", "tidyr", "yfR")

  if(sum(as.numeric(!pacotes %in% installed.packages())) != 0) {
    instalador <- pacotes[!pacotes %in% installed.packages()]
    for(i in 1:length(instalador)) {
      install.packages(instalador, dependencies = T)
      break()}
    sapply(pacotes, require, character = T)
  } else {
    sapply(pacotes, require, character = T)

  remove(pacotes)

  # get stock data from yahoo finances
  data <-
    yf_get(
      tickers = stock_name, 
      bench_ticker = origin,
      first_date = start_date,
      last_date = end_date) %>%
    select(date = ref_date, price = price_close) %>%
    arrange()

  print(head(data))

  ggplotly(
    data %>%
      ggplot() + 
      geom_line(aes(x = date, y = price, color="Price"), size=0.3) + 
      geom_smooth(method = lm, aes(x = date, y = price, color="Trend"), se = F, size = 0.5) +
      labs(title="General view", 
           x = "Period",
           y = "Price") +
      theme_minimal())

  # as we are working with time series, it necessary set a cut point to spare training and test data according with the 
  # percentage defined by the user, the default value will be 80%
  cut_point <-
    round(training_data_percentage * nrow(data))

  # --- data prep

  # as we observe the data we can see a trend of downing or growing and this kind of phenomenon reduce the acuracy of the model 
  # and the capacity of the reunal net make acertive predictions. To solve that problem we need to remove that trend, by using the
  # difference between the values and they previus value
  data_without_trend <-
    data %>%
    #lag function return the previews value
    mutate(diff = price - lag(price)) %>%
    #as the first line don´t have a previus value it shuld be removed
    filter(row_number() >= 2) %>%
    mutate(type = ifelse(row_number() >= cut_point, "test", "training"))  

  print(head(data_without_trend))

  # the next step is to create a dataframe with some values to be used on the training data
  scale_factors  <-
    data_without_trend %>%
    filter(type == "training") %>%
    summarise(min_diff = min(diff), 
              max_diff = max(diff),
              diff_amplitude = max_diff - min_diff,
              diff_mean = mean(diff),
              diff_sd = sd(diff))

  # now the data need to be normalized in -1 until 1 scale where all values will be replaced with a 
  # proportional value on that scale
  normalized_data <-
    data_without_trend %>%
      select(date, price, diff, type) %>%
      mutate(
        standard_diff = 2 * ((diff - scale_factors$min_diff) / scale_factors$diff_amplitude - .5))  

  ggplotly(
    normalized_data %>%
      ggplot() +
      geom_line(aes(x = date, y = standard_diff, color = type), size = 0.3) +
      geom_smooth(
        method = lm,
        aes(x = date, y = standard_diff, color = "Trend"), se = F, size = 0.2) +
      labs(title = "Normalized Data, without trend \n in -1 to 1 scale", 
           x = "Período",
           y = "Cotação") +
      theme_minimal()
  )

  # --- model data prep

  # to prepare the data is necessary a second lag, but this time on the variable that store the 
  # difference between the stock price with the previous day
  model_normalized_data <-
    normalized_data %>%
    # Like mutate function, transmute can create new columns and return others that we want
    transmute(date, y_var = standard_diff, x_var = lag(y_var), type) %>% 
    # removing NA values
    na.omit()

  print(head(model_normalized_data))

  # training data
  y_training <- model_normalized_data %>% filter(type == "training") %>% select(y_var) %>% as.matrix()
  x_training <- model_normalized_data %>% filter(type == "training") %>% select(x_var) %>% as.matrix()
  
  # test data, to evaluate the model accuracy
  y_test <- model_normalized_data %>% filter(type == "test") %>% select(y_var) %>% as.matrix()
  x_test <- model_normalized_data %>% filter(type == "test") %>% select(x_var) %>% as.matrix()  

  # --- model config

  # so we can use the Keras library to create a time series is necessary transform que x training 
  # variable in a matrix of 3 dimentions where the first dimention should be the size of the 
  # variable and the others 1

  dim(x_training) <- c(length(x_training), 1, 1)

  #lstm_model will store the neural net model with two layers , both using 50 entries.
  #batch_input_shape determinates the format of how the data will move in the neural net, 
  #on this case a 3 dimentions matrix
  #stateful indicate to the model generator the state of the last sample processed and if this 
  #sample should be used on the next sample processed
  lstm_model <-
    keras_model_sequential() %>%
    layer_lstm(units = 50,
               batch_input_shape = c(1, 1, 1),
               return_sequences = T,
               stateful =  T,
               dropout = 0.05) %>%
    layer_lstm(units = 50,
               return_sequences = TRUE,
               stateful = TRUE) %>%
    layer_dense(units = 1)

  # the compile command indicates the next model configurations as the learning rate and the drop
  # rate, this will avoid eventuals overffitings where the model becomes so complex that  
  # lost his hability to predict new data. The last parameter indicates the metric that will be 
  # used evaluate the increase rate on each iteration, on this case MSE
  lstm_model %>%
    compile(
      loss = "mean_squared_error",
      optimizer = optimizer_adam(learning_rate = 0.05, decay = 1e-6),
      metrics = c("mse"))
  
  print("fitting...")
  
  # creation model process
  lstm_model %>%
    fit(x_training,
        y_training,
        epochs = lstm_executions,
        batch_size = 1,
        verbose = 1,
        shuffle = F)

  print("predicting...")

  # prediction process
  lstm_prediction <-
    lstm_model %>%
    predict(x_test, batch_size = 1)

  # as the data needed to be scaled before the model creation the predicted values 
  # also need to be scaled
  lstm_prediction <-
    ((lstm_prediction / 2 + .5) * scale_factors$diff_amplitude) +
    scale_factors$min_diff

  predictions_df <-
    normalized_data %>% 
    # only the test data will be used
    filter(type == "test") %>%
    # the predicted value should be added to the price value
    mutate(lstm_prediction_value = lstm_prediction + price)

  lstm_predictions_plot <-
    predictions_df %>%
    select(date, price, lstm_prediction_value) %>%
    pivot_longer(cols = -date,
                 names_to = "price",
                 values_to = "value") %>%
    ggplot(aes(x = date, y = value, color = price)) +
    geom_line() +
    labs(title = "LSTM Prediction vs real price") +
    theme_minimal()
  plot(lstm_predictions_plot)

  lstm_mse <-
    predictions_df %>%
    group_by() %>%
    group_by() %>%
    summarise(eqm = mean((price - lstm_prediction_value)^2, na.rm = T))

  print(paste('MSE:',lstm_mse))

  return_list <-
    list(
      parameters =
        data.frame(
          stock_name = stock_name,
          start_date = start_date,
          end_date = end_date,
          arima_compare = arima_compare),
      stock_dataset = data,
      lstm_model = lstm_model,
      predictions_df =
        predictions_df %>%
        mutate(diff = price - lstm_prediction_value) %>%
        select(date, price, lstm_prediction_value, diff),
      mse = lstm_mse,
      lstm_predictions_plot = lstm_predictions_plot
    )

  if (arima_compare == T) {
    print("Starting ARIMA... ")

    arima_model <-
      auto.arima(
        y_training,
        trace = F,
        approximation = F)

    arima_predictions <-
      forecast(arima_model, h = length(y_test))

    predictions_df <-
      predictions_df %>%
      mutate(arima_prediction_value =
               price +
               ((arima_predictions$mean / 2 + .5) * scale_factors$diff_amplitude) +
               scale_factors$min_diff)

    arima_predictions_plot <-
      predictions_df %>%
      select(date, price, arima_prediction_value, lstm_prediction_value) %>%
      ggplot() +
      geom_line(aes(y = lstm_prediction_value, x = date, color = "lstm_prediction_value")) +
      geom_line(aes(y = arima_prediction_value, x = date, color = "arima_prediction_value")) + 
      geom_line(aes(y = price, x = date, color = "price")) +
      labs(title="ARIMA Prediction vs LSTM and real price") +
      theme_minimal()

    plot(arima_predictions_plot)

    arima_mse <-
      predictions_df %>%
      group_by() %>%
      summarise(lstm = mean((price - lstm_prediction_value) ^ 2,na.rm = T),
                arima = mean((price - arima_prediction_value)^2,na.rm = T)) %>%
      melt()

    print(arima_eqm)


    models_eqm_plot <-
      predictions_df %>% 
      summarise(lstm = mean((price - lstm_prediction_value) ^ 2, na.rm = TRUE),
                arima = mean((price - arima_prediction_value)^2, na.rm = TRUE)) %>%
      melt() %>%
      ggplot(aes(y = value, x = variable, fill = factor(variable))) +
      geom_bar(stat = "identity") + 
      geom_label(aes(label = (round(value, 3))), hjust = 0, color = "white", size = 5) +
      coord_flip() +
      labs(title = "MSE comparation",
           x = "models",
           y = "") +
      theme_minimal()
    plot(models_eqm_plot)

    return_list[["arima_model"]] <- arima_model
    return_list[["predictions_df"]] <-
      predictions_df %>%
      mutate(lstm_diff = price - lstm_prediction_value,
             arima_diff = price - arima_prediction_value) %>%
      select(date, price, lstm_prediction_value, lstm_diff, arima_prediction_value, arima_diff)
    return_list[["mse"]] <- arima_mse
    return_list[["arima_predictions_plot"]] <- arima_predictions_plot
    return_list[["models_eqm_plot"]] <- models_eqm_plot
  } 

  print("-----------------------------------")
  print("Execution completed")
  print(return_list["parameters"])
  print("MSE")
  print(return_list[["eqm"]])

  return(return_list)
}

return_petr4 <- 
    lstm_stock(origin = '^BVSP', stock_name = 'PETR4.SA', start_date = '2015-01-01', lstm_executions = 100, arima_compare = T)
return_mglu3 <- 
    lstm_stock(origin = '^BVSP', stock_name = 'MGLU3.SA', start_date = '2015-01-01', lstm_executions = 100, arima_compare = T)

return_ntco3 <- 
    lstm_stock(
        origin = '^BVSP', 
        stock_name = 'GGBR4.SA', 
        start_date = '2015-01-01', 
        lstm_executions = 100, 
        arima_compare = T)