clc
clear all
close all

%% Prepare input data

%%% ALL SIGNALS ARE RESAMPLED AT 4 kHz
load Healthy_patients_shallow
load COVID_patients_shallow
% load Healthy_patients_deep
% load COVID_patients_deep

load('Final_information.mat');

%%% Change this accordingly to deep when using the deep dataset
Healthy_patients = Healthy_patients_shallow;
COVID_patients = COVID_patients_shallow;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Labels
All_dx = Final_information.covid_status;
count_healthy = 0;
count_covid = 0;
healthy_id = double.empty;
covid_id = double.empty;
for i = 1:size(All_dx,1)
    selected_dx = All_dx{i,1};
    if isempty(selected_dx) == 1
       continue
    end
    check_healthy = (string(selected_dx) == "healthy");
    check_covidmild(i,1) = (string(selected_dx) == "positive_mild");
    check_covidmoderate(i,1) = (string(selected_dx) == "positive_moderate");
    check_covidasymp(i,1) = (string(selected_dx) == "positive_asymp");

    if check_healthy == 1
       count_healthy = count_healthy + 1;
       healthy_id(count_healthy,1) = i;
    end
    if (check_covidmoderate(i,1) == 1) || (check_covidmild(i,1) == 1) || (check_covidasymp(i,1) == 1)
       count_covid = count_covid + 1;
       covid_id(count_covid,1) = i;
    end       
            
end

Healthy_age = Final_information.a(healthy_id(400:522),1); 
Healthy_gender = Final_information.g(healthy_id(400:522),1); 
Healthy_all_stats = Final_information(healthy_id(400:522),1);  
Healthy_gender_edit = double.empty;
for i = 1:length(Healthy_gender)
    selected_gender = Healthy_gender{i,1};
    if string(selected_gender) == "male"
       Healthy_gender_edit(i,1) = 0;
    else
       Healthy_gender_edit(i,1) = 1;
    end
end

COVID_age = Final_information.a(covid_id,1);
COVID_gender = Final_information.g(covid_id,1);
COVID_gender_edit = double.empty;
for i = 1:length(COVID_gender)
    selected_gender = COVID_gender{i,1};
    if string(selected_gender) == "male"
       COVID_gender_edit(i,1) = 0;
    else
       COVID_gender_edit(i,1) = 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Splitting signals
Healthy_patients_increased = cell.empty;
sound_length = 64000;
limiting_length = 5000;
for i = 1:length(Healthy_patients)
    selected_data = Healthy_patients{i,1};
    segments = double.empty;
    
    if mean(isnan(selected_data)) > 0
       continue
    end
    
    if size(selected_data,1) ~= 1
       selected_data = selected_data';
    end
    
    if length(selected_data) <= limiting_length
       continue
    end
    number_of_segments = floor(length(selected_data)./sound_length);
    
    if number_of_segments == 0
       segments = [selected_data,zeros(1,sound_length-length(selected_data))];
    else
    initial_count = 0;
    for j = 1:1%number_of_segments
        segments = [segments;selected_data(initial_count+1:initial_count+sound_length)];
        initial_count = initial_count+sound_length;
    end
    end
    
    Healthy_patients_increased{i,1} = segments;
end
Dataset_healthy_final = cell.empty;
Healthy_age_final = double.empty;
Healthy_gender_final = double.empty;
count = 0;
for j = 1:length(Healthy_patients_increased)
    selected_patient = Healthy_patients_increased{j,1};
    
    for o = 1:size(selected_patient,1)
        count = count + 1;
        Dataset_healthy_final{count,1} = selected_patient(o,:)';
        Healthy_age_final(count,1) = Healthy_age(j,1);
        Healthy_gender_final(count,1) = Healthy_gender_edit(j,1);
    end
end
Dataset_healthy_final2 = cell.empty;
Healthy_age_final2 = double.empty;
Healthy_gender_final2 = double.empty;
count = 0;
for p = 1:length(Dataset_healthy_final)
    selected_signal = Dataset_healthy_final{p,1};
    selected_age = Healthy_age_final(p,1);
    selected_gender = Healthy_gender_final(p,1);
    if length(find(selected_signal == 0)) == length(selected_signal)
       continue
    else
       count = count + 1;
       Dataset_healthy_final2{count,1} =  selected_signal;
       Healthy_age_final2(count,1) = selected_age;
       Healthy_gender_final2(count,1) = selected_gender;
    end
end

COVID_patients_increased = cell.empty;
for i = 1:length(COVID_patients)
    selected_data = COVID_patients{i,1};
    segments = double.empty;
    
    if mean(isnan(selected_data)) > 0
       continue
    end
    
    if size(selected_data,1) ~= 1
       selected_data = selected_data';
    end
    
    if length(selected_data) <= limiting_length
       continue
    end
    number_of_segments = floor(length(selected_data)./sound_length);
    
    if number_of_segments == 0
       segments = [selected_data,zeros(1,sound_length-length(selected_data))];
    else
    initial_count = 0;
    for j = 1:1%number_of_segments
        segments = [segments;selected_data(initial_count+1:initial_count+sound_length)];
        initial_count = initial_count+sound_length;
    end
    end
    
    COVID_patients_increased{i,1} = segments;
end
Dataset_COVID_final = cell.empty;
COVID_age_final = double.empty;
COVID_gender_final = double.empty;
count = 0;
for j = 1:length(COVID_patients_increased)
    selected_patient = COVID_patients_increased{j,1};
    
    for o = 1:size(selected_patient,1)
        count = count + 1;
        Dataset_COVID_final{count,1} = selected_patient(o,:)';
        COVID_age_final(count,1) = COVID_age(j,1);
        COVID_gender_final(count,1) = COVID_gender_edit(j,1);
    end
end
Dataset_COVID_final2 = cell.empty;
COVID_age_final2 = double.empty;
COVID_gender_final2 = double.empty;
count = 0;
for p = 1:length(Dataset_COVID_final)
    selected_signal = Dataset_COVID_final{p,1};
    selected_age = COVID_age_final(p,1);
    selected_gender = COVID_gender_final(p,1);
    if length(find(selected_signal == 0)) == length(selected_signal)
       continue
    else
       count = count + 1;
       Dataset_COVID_final2{count,1} =  selected_signal;
       COVID_age_final2(count,1) = selected_age;
       COVID_gender_final2(count,1) = selected_gender;
    end
end

Final_dataset = [Dataset_healthy_final2;Dataset_COVID_final2];
Final_age = [Healthy_age_final2;COVID_age_final2];
Final_gender = [Healthy_gender_final2;COVID_gender_final2];
Class_healthy = ones(length(Dataset_healthy_final2),1);
Class_COVID = 2.*ones(length(Dataset_COVID_final2),1);
Final_classes = [Class_healthy;Class_COVID];

%% Extract Hand-crafted features

Final_dataset2 = cell.empty;
MFCC_dataset = cell.empty;
spectral_dataset = single.empty;
pitch_dataset = single.empty;
spectral_kurtosis = single.empty;
spectral_skewness = single.empty;
spectral_kspread = single.empty; 
spectral_centroid = single.empty; 
scalogram_dataset = cell.empty;
for i = 1:size(Final_dataset,1)
    i
    selected_signal = Final_dataset{i,1};
    [data_waveleted,c2] = wden(double(selected_signal),'modwtsqtwolog','s','mln',4,'db5');
    normalized_signal = normalize(data_waveleted');
    Final_dataset2{i,1} = single(normalized_signal);
    
    %%%%%%%%%%%%%%%%%% MFCC
    [coeffs,delta,deltaDelta,loc] = mfcc(single(normalized_signal),4000,'NumCoeffs',13,'LogEnergy','Ignore');
    mean_MFCC(1,:) = mean(coeffs);
    std_MFCC(1,:) = std(coeffs);
    kurtosis_MFCC(1,:) = kurtosis(coeffs);
    skewness_MFCC(1,:) = skewness(coeffs);
    for o = 1:size(coeffs,2)
    entropy_MFCC(1,o) = sampen(coeffs(:,o), 1, 0.2);
    end
    for o = 1:size(coeffs,2)
    HFD_MFCC(1,o) = Higuchi_FD(coeffs(:,o),5);
    end
    for o = 1:size(coeffs,2)
    KFD_MFCC(1,o) = Katz_FD(coeffs(:,o));
    end
    for o = 1:size(coeffs,2)
    spectral_entropy_MFCC(1,o) = pentropy(coeffs(:,o),4000,'Instantaneous',false);
    end
    for o = 1:size(coeffs,2)
    ZCR_MFCC(1,o) = ZCR(coeffs(:,o));
    end
    
    kurtosis_value = kurtosis(single(normalized_signal));
    skewness_value = skewness(single(normalized_signal));
    for k = 1:2
    entropy1 = sampen(single(normalized_signal(1:32000)), 1, 0.2);
    entropy2 = sampen(single(normalized_signal(32001:64000)), 1, 0.2);
    end
    entropy = (entropy1+entropy2)./2;
    HFD = Higuchi_FD(single(normalized_signal), 5); 
    KFD = Katz_FD(single(normalized_signal));
    spectral_entropy = pentropy(single(normalized_signal),4000,'Instantaneous',false);
    ZCR_value = ZCR(single(normalized_signal));

    spectral_dataset(i,:) = [mean_MFCC,std_MFCC,kurtosis_MFCC,skewness_MFCC,...
                             entropy_MFCC,HFD_MFCC,KFD_MFCC,spectral_entropy_MFCC,ZCR_MFCC,...
                             kurtosis_value,skewness_value,entropy,HFD,KFD,spectral_entropy,...
                             ZCR_value];
end

%% Deep learning 

acc = double.empty;
out_class_total = double.empty;
T_test_total = double.empty;
test_acc = double.empty;
Overall_nets = cell.empty;
testing_classes = double.empty;
conf = cell.empty;
test_idx = double.empty;
scores_NN = double.empty;
scores = double.empty;
array = double.empty;

    X_train = Final_dataset2([1:2,4:end]); %Features for training
    age_train = Final_age([1:2,4:end]);
    gender_train = Final_gender([1:2,4:end]);
    spectral_train = spectral_dataset([1:2,4:end],:);
     
    X_test  = Final_dataset2(3);   %Features test set
    age_test = Final_age(3);
    gender_test = Final_gender(3);
    spectral_test = spectral_dataset(3,:);
    
    T_train = Final_classes([1:2,4:end]);  
    T_test  = Final_classes(3);
    testing_classes = T_test;
    
    %%%%%%%%% Augmentation
    classes_aug = unique(T_train);
    Augmented_dataset = cell.empty;
    Augmented_labels = double.empty;
    for class_count = 1:length(classes_aug)

    selected_class = classes_aug(class_count,1);
    class_idx = find(T_train == selected_class);
    class_data = X_train(class_idx,1);
    labels_of_class = T_train(class_idx); 

        augmenter = audioDataAugmenter( ...
        "AugmentationMode","sequential", ...
        "NumAugmentations",floor(3000/length(class_data)), ...
        ...
        "TimeStretchProbability",0, ...
        "SpeedupFactorRange", [1,1], ...
        ...
        "PitchShiftProbability",0, ...
        ...
        "VolumeControlProbability",1, ...
        "VolumeGainRange",[-10,10], ...
        ...
        "AddNoiseProbability",0, ...
        ...
        "TimeShiftProbability",1, ...
        "TimeShiftRange", [-500e-2,500e-2]);

    all_data = cell.empty;
    for data_order = 1:length(class_data)
    data = augment(augmenter,class_data{data_order,1},4000);
    all_data = [all_data;class_data{data_order,1};data.Audio];
    end   

    aug_labels = repmat(labels_of_class(1,1),length(all_data),1);

    Augmented_dataset = [Augmented_dataset;all_data];
    Augmented_labels = [Augmented_labels;aug_labels];

    end
    
% Use only if you need weighted layer
% class_A_count = length(find(T_train==1));
% class_B_count = length(find(T_train==2));
% classes = ["1","2"];
% classWeights = 1-[class_A_count./(class_A_count+class_B_count),...
%                 class_B_count./(class_A_count+class_B_count)];

inputSize = [size(X_train{1,1},1) size(X_train{1,1},2) 1];
numHiddenUnits = 256;
numClasses = 2;

layers = [ ...
    sequenceInputLayer(inputSize,'Name','input')
    sequenceFoldingLayer('Name','fold')
    
    convolution2dLayer([9 1],16,'Stride',[8,1],'Name','conv1','Padding','same')
    maxPooling2dLayer([3 1],'Stride',[2,1],'Name','maxpool1','Padding','same')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
  
    convolution2dLayer([5 1],32,'Stride',[4,1],'Name','conv11','Padding','same')
    maxPooling2dLayer([3 1],'Stride',[2,1],'Name','maxpool11','Padding','same')
    batchNormalizationLayer('Name','bn11')
    reluLayer('Name','relu11')
    
    convolution2dLayer([3 1],64,'Stride',[2,1],'Name','conv111','Padding','same')
    maxPooling2dLayer([3 1],'Stride',[2,1],'Name','maxpool111','Padding','same')
    batchNormalizationLayer('Name','bn111')
    reluLayer('Name','relu111')
    
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    
    bilstmLayer(numHiddenUnits,'OutputMode','last','Name','bilstm')
        
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')
%     classificationLayer('Classes',classes,'ClassWeights',classWeights,'Name','classification')
    ];

lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,'fold/miniBatchSize','unfold/miniBatchSize');

% analyzeNetwork(lgraph)

maxEpochs = 2;
miniBatchSize = 32;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Verbose',false,...
    'InitialLearnRate',0.001,...
    'L2Regularization',0.000001,...
    'Shuffle','every-epoch',...
    'Plots','training-progress');

    clear net
    rng('default')
    [net info] = trainNetwork(Augmented_dataset,categorical(Augmented_labels),lgraph,options);
    [out_class,scores] = classify(net,X_test,'MiniBatchSize',miniBatchSize);
    test_acc = (sum(categorical(out_class) == categorical(T_test))./numel(categorical(T_test))).*100
                        
    %%%%%%%%%%%% Combine CNN-BiLSTM with feature layer
layer = 'bilstm';
featuresTrain = activations(net,X_train,layer);
featuresTrain = single(featuresTrain');
featuresTest = activations(net,X_test,layer);
featuresTest = single(featuresTest');

additional_features_train = [age_train,...
                             gender_train,...
                             spectral_train];
                         
additional_features_test = [age_test,...
                            gender_test,...
                            spectral_test];

[idx,fsc_scores] = fscchi2(additional_features_train,T_train);
idx_features = idx(1:20);
% figure
% bar(fsc_scores(idx_features))
% xticks([1:1:20])
% xticklabels(label_names(idx_features))
% xlabel('Predictor rank')
% ylabel('Predictor importance score')

featuresTrain = [featuresTrain,additional_features_train(:,idx_features)];
featuresTest = [featuresTest,additional_features_test(:,idx_features)];

featuresTrain = normalize(featuresTrain,2);
featuresTest = normalize(featuresTest,2);

%%%%%%%%%%%%%% NN
numFeatures = size(featuresTrain,2);
numClasses = length(unique(T_train));
miniBatchSize = 32;
maxEpochs = 50;
layers2 = [
    featureInputLayer(numFeatures,'Name','input')
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','sm')
    classificationLayer
%     classificationLayer('Classes',classes,'ClassWeights',classWeights,'Name','classification')
    ];

options = trainingOptions('adam', ...
    'ExecutionEnvironment','gpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.001, ...
    'L2Regularization',0.000001, ...
    'Shuffle','every-epoch', ...
    'Verbose',false,...
    'Plots','training-progress');

rng('default')
[net2 info] = trainNetwork(featuresTrain,categorical(T_train),layers2,options);

[out_class2,scores_NN] = classify(net2,featuresTest,'MiniBatchSize',miniBatchSize);

test_acc = (sum(categorical(out_class2) == categorical(T_test))./numel(categorical(T_test))).*100
