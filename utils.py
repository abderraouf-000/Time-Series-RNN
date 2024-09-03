def train_epoch(model, Dataloader, device , loss_function , optimizer):

    model.train();
    running_loss = 0;
    for batch_index, batch in enumerate(Dataloader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device);
        output = model(x_batch);
        loss = loss_function(output, y_batch);
        running_loss += loss;
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if(batch_index % 100 == 99):
            avg_loss = running_loss / 100;
            running_loss = 0.0;
