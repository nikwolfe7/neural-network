package main;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mlsp.cs.cmu.edu.factory.DNNFactory;
import mlsp.cs.cmu.edu.factory.FeedForwardDNNFactory;
import mlsp.cs.cmu.edu.factory.LinearOutputFeedForwardDNNFactory;
import mlsp.cs.cmu.edu.factory.SigmoidNetworkAbstractFactoryImpl;
import mlsp.cs.cmu.edu.factory.SimpleFeedForwardDNNFactory;
import mlsp.cs.cmu.edu.factory.ThresholdOutputFFDNNFactory;
import mlsp.cs.cmu.edu.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.util.DNNUtils;
import training.AdditionDataGenerator;
import training.DNNTrainingModule;
import training.DataInstance;
import training.DataInstanceGenerator;
import training.MultipleOutputAddition;
import training.XORGenerator;

public class FactoryDriver {

  static DataInstanceGenerator dataGen = new XORGenerator();
  static DNNFactory factory = new ThresholdOutputFFDNNFactory(dataGen.getNewDataInstance(), 10);
  
  public static void main(String[] args) {
    
    /* Build the network */
    NeuralNetwork net = factory.getInitializedNeuralNetwork();

    /* Generate training and test data */
    List<DataInstance> training = getData(10000);
    List<DataInstance> testing = getData(10);

    /* Train the network */
    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
    trainingModule.setOutputOn(true);
    trainingModule.doTrainNetworkUntilConvergence();

    /* Test the network */
    trainingModule.doTestTrainedNetwork();

  }

  private static List<DataInstance> getData(int numInstances) {
    List<DataInstance> data = new ArrayList<>();
    for (int i = 0; i < numInstances; i++)
      data.add(dataGen.getNewDataInstance());
    return data;
  }
}
