package main;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.FeedForwardDNNAbstractFactory;
import mlsp.cs.cmu.edu.dnn.factory.LinearOutputFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ReadSerializedFileDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkElementFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ThresholdOutputFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.AdditionDataGenerator;
import mlsp.cs.cmu.edu.dnn.training.CosineGenerator;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceGenerator;
import mlsp.cs.cmu.edu.dnn.training.MultipleOutputAddition;
import mlsp.cs.cmu.edu.dnn.training.XORGenerator;
import mlsp.cs.cmu.edu.dnn.util.BinaryThresholdOutput;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;

public class CosineFactoryDriver {

  static String sep = System.getProperty("file.separator");
  static DataInstanceGenerator dataGen = new CosineGenerator();
  static DNNFactory factory = new SigmoidNetworkFFDNNFactory(dataGen.getNewDataInstance(),50,50);
//  static DNNFactory factory = new ReadSerializedFileDNNFactory("cos.network.dnn");
  
  public static void main(String[] args) {
    
    /* Build the network */
    NeuralNetwork net = factory.getInitializedNeuralNetwork();

    /* Generate training and test data */
    List<DataInstance> training = getData(1000000);
    List<DataInstance> testing = getData(1000);

    /* Train the network */
    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
    trainingModule.setOutputOn(true);
//    trainingModule.setOutputAdapter(new BinaryThresholdOutput());
    trainingModule.setConvergenceCriteria(1.0e-8, -1, 1, 1000);
    trainingModule.doTrainNetworkUntilConvergence();

    /* Test the network */
    trainingModule.doTestTrainedNetwork();
    trainingModule.saveNetworkToFile("models" + sep +"cos.big.network.dnn");
  }

  private static List<DataInstance> getData(int numInstances) {
    List<DataInstance> data = new ArrayList<>();
    for (int i = 0; i < numInstances; i++)
      data.add(dataGen.getNewDataInstance());
    return data;
  }
}
