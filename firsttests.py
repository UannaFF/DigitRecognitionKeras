   ########## We starst with the first test for just one hidden layer 32 nodes

    firstmodel = model(10, num_classes, X_train.shape[1], 'adam')

    #firstmodel.summary() #print summary of model

    firstmodel.save_weights("./weights.txt")
    #use_model.load_weights(filepath, by_name=False)

    # Fit the model
    history = firstmodel.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, 
        shuffle=False)

    plotHistory(history)

    # evaluate the model
    stats = firstmodel.evaluate(X_train, Y_train)

    print("stats first model: "+str(stats))

    # calculate predictions
    predictions = firstmodel.predict_classes(X_test, verbose=0)

    getFailCases(predictions, Y_test)

    ################## Finish test

    ################## Tests for first model with more nodes

    firstmodel = model(32, num_classes, X_train.shape[1], 'adam')

    #firstmodel.summary() #print summary of model
    #firstmodel.load_weights("./weights.txt", by_name=False)

    # Fit the model
    history = firstmodel.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, 
        shuffle=False)

    plotHistory(history)

    # evaluate the model
    stats = firstmodel.evaluate(X_train, Y_train)

    print("stats first model: "+str(stats))

    # calculate predictions
    predictions = firstmodel.predict_classes(X_test, verbose=0)
    #print(str(predictions))

    getFailCases(predictions, Y_test)

    ######### Finish test

    ################## Tests for first model with more nodes

    secondmodel = model2(32, num_classes, X_train.shape[1], 'adam')

    #firstmodel.summary() #print summary of model
    #seconmodel.load_weights("./weights.txt", by_name=False)

    # Fit the model
    history = secondmodel.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, 
        shuffle=False)

    plotHistory(history)

    # evaluate the model
    stats = secondmodel.evaluate(X_train, Y_train)

    print("stats first model: "+str(stats))

    # calculate predictions
    predictions = secondmodel.predict_classes(X_test, verbose=0)
    #print(str(predictions))

    getFailCases(predictions, Y_test)

    ######### Finish test