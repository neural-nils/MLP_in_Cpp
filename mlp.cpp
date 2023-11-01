
#include <fstream>
#include <random>
#include <string>
#include <vector>


//as written on Yann Lecuns webpage (http://yann.lecun.com/exdb/mnist/)
//the bytes are stored in big-endian; therefore we need to flip the bytes
//tutorial on byte order from IBM: https://developer.ibm.com/articles/au-endianc/
int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4; 
    c1 = i & 255;
    c2 = (i >> 8) & 255; 
    c3 = (i >> 16) & 255; 
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4; 
}

class DataLoader {
    public:
        DataLoader(const std::string filename, const std::string filename_labels);
        bool isEof(void){return m_inputDataFile.eof();}    
        unsigned getNextInputs(std::vector<double> &inputVals);
        int getTargetOutputs(std::vector<double> &targetOutputVals);
        int return_num_of_images() {return p_number_of_images;};

        //if you are interested in the MNIST dataset; This prints some information.
        void printHeaderInformation(){
            printf("\n");        
            printf("Images:\n");
            printf("p_magic_number: %d\n", p_magic_number);        
            printf("p_number_of_images: %d\n", p_number_of_images);        
            printf("p_n_rows: %d\n", p_n_rows);
            printf("p_n_cols: %d\n", p_n_cols);        
            printf("\n");
            printf("Labels:\n");
            printf("p_magic_number_label: %d\n", p_magic_number_label);
            printf("p_number_of_labels: %d\n", p_number_of_labels);
            printf("\n");
        }
    
    private:
        std::ifstream m_inputDataFile;
        std::ifstream m_inputDataFile_labels;

        int p_magic_number_label; 
        int p_number_of_labels; 
        int p_magic_number;
        int p_number_of_images; 
        int p_n_rows;
        int p_n_cols;
};


DataLoader::DataLoader(const std::string filename, const std::string filename_labels){
    m_inputDataFile.open(filename.c_str());
    m_inputDataFile_labels.open(filename_labels.c_str());

    if(m_inputDataFile.is_open())
    {
        p_magic_number = 0; 
        p_number_of_images = 0; 
        p_n_rows = 0; 
        p_n_cols = 0;
        m_inputDataFile.read((char*)&p_magic_number, sizeof(p_magic_number));
        p_magic_number = reverseInt(p_magic_number);
        m_inputDataFile.read((char*)&p_number_of_images, sizeof(p_number_of_images));
        p_number_of_images = reverseInt(p_number_of_images);
        m_inputDataFile.read((char*)&p_n_rows, sizeof(p_n_rows));
        p_n_rows=reverseInt(p_n_rows);
        m_inputDataFile.read((char*)&p_n_cols, sizeof(p_n_cols));
        p_n_cols = reverseInt(p_n_cols);

        m_inputDataFile_labels.read((char*)&p_magic_number_label, sizeof(p_magic_number_label));
        p_magic_number_label = reverseInt(p_magic_number_label);
        m_inputDataFile_labels.read((char*)&p_number_of_labels, sizeof(p_number_of_labels));        
        p_number_of_labels = reverseInt(p_number_of_labels);
    } else {
        printf("there is a problem\n");
    }

}

unsigned DataLoader::getNextInputs(std::vector<double> &inputVals)
{    
    inputVals.clear();
    unsigned char temp = 0;     
    for (int i = 0; i < 28*28; ++i)
    {        
        m_inputDataFile.read((char*)&temp, sizeof(char));       
        inputVals.push_back(((double)temp)/255);         
    }
    
    double mean = 0;
    for (int i = 0; i < inputVals.size(); ++i)
    {
        mean += inputVals[i];
    }
    mean = mean / inputVals.size();

    for (int i = 0; i < inputVals.size(); ++i)
    {
        inputVals[i] = inputVals[i] - mean;        
    }

    return inputVals.size();
}


int DataLoader::getTargetOutputs(std::vector<double> &targetOutputVals){
    targetOutputVals.clear();
    for (int i = 0; i < 10; ++i)
    {
        targetOutputVals.push_back((double) 0);
    }
    char temp = 0;     
    m_inputDataFile_labels.read((char*)&temp, sizeof(temp));            
    int result = (int)temp;
    targetOutputVals[result] = 1;    
    return result;
}


class Neuron
{
    public:
    Neuron(int weights)
    {
        for (int i = 0; i < weights; ++i)
        {            
            double r1 = rand() % 10;
            double r2 = rand() % 10;
            double r = (r1 - r2)/100;        
            m_weights.push_back(r);                                  
        }

        double r1 = rand() % 10;
        double r2 = rand() % 10;
        double r = (r1 - r2)/100; //between - 0.1 and 0.1
        m_bias = r; 
    }

    void set_activation(double i)
    {        
        m_activation = i;        
    }

    double get_activation()
    {
        return m_activation;
    }

    void forward_prop(std::vector<Neuron>& prevLayer)
    {
        double z = 0; 
        for (int neuron = 0; neuron < prevLayer.size(); ++neuron)
        {                                    
            z += (prevLayer[neuron].get_activation() * this->m_weights[neuron]);            
        }
        z += m_bias;
        this->m_z = z; 
        m_activation = activation_function(z);                        
    }

    void forward_prop_lastlayer(std::vector<Neuron>& prevLayer)
    {
        double sum = 0; 

        for (int neuron = 0; neuron < prevLayer.size(); ++neuron)
        {                                    
            sum += (prevLayer[neuron].get_activation() * this->m_weights[neuron] + m_bias);            
        }

        //important:
        //we can't calculate softmax already here, we need to have all activations of that layer
        m_activation = sum;        
    }

    double activation_function(double z)
    {
        //sigmoid:                
        //return 1/(1+exp(-z));
        //tanh:
        //return tanh(z);
        //relu:
        if (z < 0) { return 0; }
        else { return z; }        
    }

    double activation_function_derivative(double z)
    {
        //sigmoid:                
        //return 1/(1+exp(-z)) * (1 - 1/(1+exp(-z)));
        //tanh:
        //return (1 - (tanh(z)*tanh(z)));
        //relu:
        if (z < 0) { return 0; }
        else { return 1; }
    }

    void set_gradient(double g) { m_gradient = g; }

    void calculate_hidden_gradients(std::vector<Neuron>& nextLayer, int index, double derivative)
    {        
        double delta = 0;
        for (int n = 0; n < nextLayer.size(); ++n)
        {
            double weight = nextLayer[n].get_weight(index);
            double delta = nextLayer[n].get_delta();
            delta += weight*delta;
        }
        delta *= derivative;
        this->m_delta = delta;         
    }

    void set_delta(double d){ m_delta = d;}
    double get_delta() { return m_delta; }
    double get_weight(int i) { return m_weights[i];}    
    void set_weight(int i, double new_weight) { m_weights[i] = new_weight;}
    double get_z() { return m_z; }
    double get_bias() { return m_bias; }
    void set_bias(double new_bias) { m_bias = new_bias; }

    std::vector<double> get_prev_activations(std::vector<Neuron>& prevLayer)
    {
        std::vector<double> prev_activations;
        for (int i = 0; i < prevLayer.size(); ++i)
        {
            prev_activations.push_back(prevLayer[i].get_activation());
        }
        return prev_activations;
    }

    private:
    std::vector<double> m_weights;
    double m_activation;
    double m_bias; 
    double m_gradient; 
    double m_delta;
    double m_z;
};

class Net
{
    public:
    Net(std::vector<int>topology)
    {
        printf("Input is: %d\n", topology[0]);

        for (int i = 0; i < topology.size(); ++i)
        {            
            printf("\n");
            printf("Layer %d -> ", i);
            std::vector<Neuron> layer;                        
            for (int j = 0; j < topology[i]; ++j)
            {
                printf("N%d,", j);
                Neuron n(topology[i-1]);
                layer.push_back(n);
            }            
            printf("\n");
            m_net.push_back(layer);
        }

    }
    void forward_prop(const std::vector<double>& input)
    {
        for (int i = 0; i < input.size(); ++i)
        {            
            m_net[0][i].set_activation(input[i]);
        }

        //forward propagation, except the last layer
        for (int layer = 1; layer < m_net.size() -1 ; ++layer) 
        {
            std::vector<Neuron>& prevLayer = m_net[layer - 1];
            for (int neuron = 0; neuron < m_net[layer].size(); ++neuron)
            {
                m_net[layer][neuron].forward_prop(prevLayer);
            }            
        }

        //forward propagation last layer
        std::vector<Neuron>& prevLayer = m_net[m_net.size()-2];
        std::vector<Neuron>& lastLayer = m_net[m_net.size()-1];
               
        for (int neuron = 0; neuron < m_net.back().size(); ++neuron)
        {
            lastLayer[neuron].forward_prop_lastlayer(prevLayer);
        } 
    }

    //logit shift for numerical stability //optional
    void shift_output_activations()
    {
        double largest = 0; 
        for (int i = 0; i < m_net.back().size(); ++i)
        {            
            if (m_net.back()[i].get_activation() > largest)
                largest = m_net.back()[i].get_activation();            
        }                

        for (int i = 0; i < m_net.back().size(); ++i)
        {            
            double current = m_net.back()[i].get_activation();
            m_net.back()[i].set_activation((current-largest));
        }        
    }

    void get_softmax_y_hat(std::vector<double>& y_hat) {
        y_hat.clear();
        std::vector<Neuron> output_layer = m_net.back();
        double sum = 0; 
        for (int neuron = 0; neuron < output_layer.size(); ++neuron)
        {            
            sum += exp(output_layer[neuron].get_activation());
        }
        
        for (int neuron = 0; neuron < output_layer.size(); ++neuron)
        {
            double result = exp(output_layer[neuron].get_activation())/sum;
            y_hat.push_back(result);
        }
    }

    double calculate_gradients(std::vector<double>& y, std::vector<double>& y_hat){
        
        if (y.size() != y_hat.size())
        {
            printf("std::vectors don't match...\n");
        } 
        
        std::vector<double> gradients; 
        double loss = 0; 
        for (int i = 0; i < y_hat.size(); ++i)
        {            
            loss += -log(y_hat[i])*y[i];
            double delta = y_hat[i]-y[i];
            m_net.back()[i].set_delta(delta);                        
        }
        loss = loss/y_hat.size();
        
        //hidden_gradients
        for (int layer = m_net.size() - 2; layer > 0; --layer)
        {
            std::vector<Neuron>& hiddenLayer = m_net[layer];
            std::vector<Neuron>& nextLayer = m_net[layer+1];

            for (int n = 0; n < hiddenLayer.size(); ++n)
            {
                double derivative = hiddenLayer[n].activation_function_derivative(hiddenLayer[n].get_z());
                hiddenLayer[n].calculate_hidden_gradients(nextLayer, n, derivative);
            }
        }

        return loss;
    }

    void backprop(double alpha)
    {        
        for (int layer = 1; layer < m_net.size(); ++layer) 
        {
            for (int n = 0; n < m_net[layer].size(); ++n)
            {
                std::vector<Neuron> prevLayer = m_net[layer - 1];                
                std::vector<double> prev_activations = m_net[layer][n].get_prev_activations(prevLayer);
                                
                for(int w = 0; w < prev_activations.size(); ++w)
                {
                    double dw = prev_activations[w] * m_net[layer][n].get_delta();
                    double new_weight = m_net[layer][n].get_weight(w) - alpha * dw;
                    m_net[layer][n].set_weight(w, new_weight);
                }                                
                double db = m_net[layer][n].get_delta();                
                double new_bias = m_net[layer][n].get_bias() - alpha * db;
                m_net[layer][n].set_bias(new_bias);
            }
        }
    }

    private:
    std::vector<std::vector<Neuron>> m_net;
};



int main()
{            
    std::vector<int>topology{28*28, 250, 10}; 
    int epochs = 10; 

    srand(time(NULL));
    printf("initialize network according to topology...\n\n");
    Net net(topology);  
        
    printf("---------------------------------\n");    
    printf("start training...\n");

    for (int e = 0; e < epochs; ++e)
    {
        printf("epoch %d\n", e);    
        DataLoader trainData("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte");
        //trainData.printHeaderInformation();
        int count = 0; 
        int correct = 0;     
        double avg_loss = 0;     
        
        while(count < trainData.return_num_of_images())
        {               
            std::vector<double>input, y, y_hat;            
            trainData.getNextInputs(input);
            int current_target = trainData.getTargetOutputs(y);
            
            net.forward_prop(input);
            
            //if you want to experiment you can disable the output activation shift            
            net.shift_output_activations();  
            
            net.get_softmax_y_hat(y_hat);
            double largest = 0; 
            int largest_idx = 0; 
            for (int i = 0; i < y_hat.size(); ++i)
            {                
                if (y_hat[i] > largest)
                {
                    largest = y_hat[i];
                    largest_idx = i;
                }
            } 
            if (largest_idx == current_target)
            {                
                correct += 1; 
            }
            
            double loss = net.calculate_gradients(y, y_hat);        
            avg_loss += loss; 
            if (count % 1000 == 0 && count != 0)
            {
                printf("Image %d\t\t LOSS: %f, LOSS(avg): %f\n", count, loss, avg_loss/count);
            }
            
            net.backprop(0.01);            
            ++count;
        }

        double accuracy = (double)correct/(double)count;
        printf("train accuracy: %d/%d = %f\n", correct, count, accuracy);
    }

    
    printf("Evaluating test set...\n");
    DataLoader testData("MNIST/t10k-images-idx3-ubyte", "MNIST/t10k-labels-idx1-ubyte");

    int count = 0; 
    int correct = 0;     
    double avg_loss = 0;     

    while(count < testData.return_num_of_images())
    {                   
        std::vector<double>input, y, y_hat;        
        testData.getNextInputs(input);
        int current_target = testData.getTargetOutputs(y);

        net.forward_prop(input);
        net.shift_output_activations();
        
        net.get_softmax_y_hat(y_hat);
        double largest = 0; 
        int largest_idx = 0; 
        for (int i = 0; i < y_hat.size(); ++i)
        {            
            if (y_hat[i] > largest)
            {
                largest = y_hat[i];
                largest_idx = i;
            }
        } 
        if (largest_idx == current_target)
        {            
            correct += 1; 
        }
        ++count;
    }
    
    double accuracy = (double)correct/(double)count;
    printf("accuracy on test set: %d/%d = %f\n", correct, count, accuracy);

    return 0;
}
