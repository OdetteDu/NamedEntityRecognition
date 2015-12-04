package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {

	if (args.length < 2) {
	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev"); // ../data -> ./data/ in our project
	    return;
	}	    

	// this reads in the train and test datasets
	List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
	List<Datum> testData = FeatureFactory.readTestData(args[1]);	

	
	//	read the train and test data
	FeatureFactory.initializeVocab("./data/vocab.txt");
	SimpleMatrix allVecs = FeatureFactory.readWordVectors("./data/wordVectors.txt");
	allVecs.printDimensions();
	
	// initialize model 
	WindowModel model = new WindowModel(5, 100, 0.003, 10);
	model.initWeights();
	model.train(trainData);
	model.test(trainData);
	model.test(testData);

	
    }
}