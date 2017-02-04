/**
* Date:  2016
* Author: Rafael Muñoz Salinas
* Description: demo application of DBoW3
* License: see the LICENSE.txt file
*/

#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"
#include <map>

using namespace DBoW3;
using namespace std;
using namespace cv;


//vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") ;

cv::Mat im;
VideoCapture cap(0);
vector <cv::Mat> all_images;

//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;  else  return ( argv[  idx+1] ); }};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
    //loadFeatures();
}

vector<string> readImagePaths(int argc,char **argv,int start){
    vector<string> paths;
    for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
        return paths;
        
}

vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor="") throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create();
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector = cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;

    cout << "Extracting  features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
        all_images.push_back(image);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        cout<<"done detecting features"<<endl;
    }
    return features;
}

// ----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void creatVoc(const vector<cv::Mat> &features,DBoW3::Vocabulary &voc)
{
    //cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "... done!" << endl;
}


void testVocCreation(const vector<cv::Mat> &features,DBoW3::Vocabulary &voc,vector <cv::Mat> &all_images)
{

    cout<<"voc size is : "<<voc.size()<<endl;
    map<float,int> img_scores;
    cap>>im;  //get image from camera

    imshow("org",im);
    waitKey(10);
    if(im.empty())throw std::runtime_error("Could not open im");

    cv::Ptr<cv::Feature2D> fdetector_test;
    fdetector_test=cv::ORB::create();
    //fdetector_test= cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);

    vector<cv::Mat>    features_test;
    cout << "Extracting  features..." << endl;
    vector<cv::KeyPoint> keypoints_test;
    cv::Mat descriptors_test;

    cout<<"extracting features"<<endl;
    fdetector_test->detectAndCompute(im, cv::Mat(), keypoints_test, descriptors_test);
    features_test.push_back(descriptors_test);

    // lets do something with this vocabulary
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;

    for(size_t i = 0; i < features_test.size(); i++)
    {
        cout << "test i is :  "<<i << endl;
        voc.transform(features_test[i], v1);
        cout << "test i is :  "<<i << endl;
        for(size_t j = 0; j < features.size(); j++)
        {
            voc.transform(features[j], v2);

            double score = voc.score(v1, v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;

            img_scores[score] = j;


        }
    }

    cout<<"img_scores size is :"<<img_scores.size()<<endl;
    map<float , int>::const_iterator it = img_scores.end();
    //for (it = img_scores.begin(); it != img_scores.end(); ++it)
       //cout << it->first << "=" << it->second << endl;

    //cout << endl;
    --it;
    cout << it->first << "=" << it->second << endl;  //the last is the max key mean the max score

    //if(it->first>=0.3)
    {
        imshow("answer",all_images[it->second]);
        waitKey(10);
    }
    //else
    {
        cout<<"cannot recognise book ......"<<endl;
    }


    // save the vocabulary to disk
    //cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
  // cout << "Done" << endl;
}

////// ----------------------------------------------------------------------------

void testDatabase(const  vector<cv::Mat > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Vocabulary voc("small_voc.yml.gz");

    Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;

    // once saved, we can load it again
    cout << "Retrieving database once again..." << endl;
    Database db2("small_db.yml.gz");
    cout << "... done! This is: " << endl << db2 << endl;
}

//-------------------------------------------------------------------------------
void testOurSelf(vector< cv::Mat>  &features )
{

    cout<<"begin load voc"<<endl;
    Vocabulary voc_test("small_voc.yml.gz");
    cout << "finish load voc... " << endl;

    //= cv::imread("/home/gsh/libs/DBow3/build/utils/test.jpg",0);
    cap>>im;
    if(im.empty())throw std::runtime_error("Could not open im");

    cv::Ptr<cv::Feature2D> fdetector_test;
    fdetector_test=cv::ORB::create();

    vector<cv::Mat>    features_test;
    cout << "Extracting  features..." << endl;
    vector<cv::KeyPoint> keypoints_test;
    cv::Mat descriptors_test;

    cout<<"extracting features"<<endl;
    fdetector_test->detectAndCompute(im, cv::Mat(), keypoints_test, descriptors_test);
    features_test.push_back(descriptors_test);

    BowVector v1, v2;

    voc_test.transform(features_test[0], v1);
    for(size_t j = 0; j < features.size(); j++)
    {
        voc_test.transform(features[j], v2);
        double score = voc_test.score(v1, v2);
        cout << "Image " << 0 << " vs Image " << j << ": " << score << endl;
    }

}

// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
      CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<=2){
            cerr<<"Usage:  descriptor_name    image0 image1 ... \n\t descriptors:brisk,surf,orb ,akaze(only if using opencv 3)"<<endl;
            return -1;
        }

        string descriptor=argv[1];

        auto images = readImagePaths(argc,argv,2);

        vector< cv::Mat>  features = loadFeatures(images,descriptor);
        //features_test = loadFeatures(images,descriptor);


        // branching factor and depth levels
        const int k = 15;
        const int L = 5;
        const WeightingType weight = TF_IDF;
        const ScoringType score = L1_NORM;
        DBoW3::Vocabulary voc(k, L, weight, score);

        creatVoc(features,voc);
        if(!cap.isOpened())
        {
            cout << "Cannot open video/camera!" << endl;
            return 0;
        }
        //cap >> frame;

        while(1)
        {
            testVocCreation(features,voc,all_images);
        }

        cout<<"finish save voc"<<endl;


        testOurSelf(features);
        //testDatabase(features);

    }catch(std::exception &ex)
    {
        cerr<<ex.what()<<endl;
    }
    return 0;
}
