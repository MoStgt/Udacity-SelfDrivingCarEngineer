/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  
  //Initialize number of particles
  num_particles = 20;  // TODO: Set the number of particles
  default_random_engine gen;
  
  //Set standard deviations
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  //Create normal distributions
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  //Generate particles from normal distribution using GPS data where GPS data are the means
  for (int i=0; i<num_particles;i++){
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    
    particles.push_back(particle);
    
  }
  //Filter is initialized
  is_initialized = true;
  
  

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  //Set standard deviations
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  
  //Create normal distributions
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);
  
  //Calculate prediction step for each particle
  for (int i=0; i<num_particles; i++){
    double theta = particles[i].theta;
    
    //if yaw_rate approximately zero take equations for yaw rate=0 otherwise take equations for yaw rate not equal zero
    if ( fabs(yaw_rate) < 0.0001){
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);      
    } else {
      particles[i].x += velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) );
      particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) );
      particles[i].theta += yaw_rate * delta_t;
    }
    
    //Adding some noise for each particle
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  //go through each observation
  for (unsigned int i = 0; i < observations.size(); i++){    
    double obs_x = observations[i].x;
	double obs_y = observations[i].y;
  
    //Initialize min distance as large number
    double minDistance = numeric_limits<double>::max();

    //Initialize map id
    int mapID = -1;
    
    //Go through each prediction
    for (unsigned int j=0; j<predicted.size(); j++){
      
      double pred_x = predicted[j].x;
	  double pred_y = predicted[j].y;
      int pred_id = predicted[j].id;
      
      //calculate distance between observation i and predicted j
	  double current_distance = dist(obs_x, obs_y, pred_x, pred_y);
      
      //Check if calculated current distance is smaller than minDistance if yes update mapID and minDistance
      if (current_distance < minDistance){
        minDistance = current_distance;
        mapID = pred_id;
      }
    }
	observations[i].id = mapID;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double weight_normalizer = 0.0;
  for (int i = 0; i<num_particles; i++){
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;
    
    //Find landmarks in sensor range
    vector<LandmarkObs> landmarksInRange;
    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); j++){
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      if ( (fabs(particle_x - landmark_x) <= sensor_range) && (fabs(particle_y - landmark_y) <= sensor_range)){
        landmarksInRange.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }    
    }
              
    //Transform from vehicle to map coordinates
    vector<LandmarkObs> transObservations;
    for (unsigned int j=0; j<observations.size(); j++){
      LandmarkObs transOb;
      transOb.id = j;
      transOb.x = particle_x + (cos(particle_theta) * observations[j].x) - (sin(particle_theta) * observations[j].y);
      transOb.y = particle_y + (sin(particle_theta) * observations[j].x) + (cos(particle_theta) * observations[j].y);
      transObservations.push_back(transOb);            
    }
          
    
    //Associate landmark to observation
    dataAssociation(landmarksInRange, transObservations);
          
    //Reinit weight
    particles[i].weight = 1.0;
          
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
          
    for (unsigned int k = 0; k<transObservations.size(); k++){
      double transObservations_x = transObservations[k].x;
      double transObservations_y = transObservations[k].y;
      double transObservations_id = transObservations[k].id;
      
      
      for (unsigned int l = 0; l<landmarksInRange.size(); l++){
        double landmarksInRange_x = landmarksInRange[l].x;
        double landmarksInRange_y = landmarksInRange[l].y;
        double landmarksInRange_id = landmarksInRange[l].id;
        
        if (transObservations_id == landmarksInRange_id){
          double mvgp = (1.0/(2.0 * M_PI * sig_x * sig_y))* exp(-1.0 * ((pow((transObservations_x - landmarksInRange_x), 2)/(2.0 * sig_x*sig_x)) + (pow((transObservations_y - landmarksInRange_y), 2)/(2.0 * sig_y*sig_y))));
          particles[i].weight *= mvgp;
        }
      }
    }
    weight_normalizer += particles[i].weight;
  }        
  //Normalize weights of all particles
  for (unsigned int i = 0; i<particles.size(); i++){
    particles[i].weight /= weight_normalizer;      
    }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  default_random_engine gen;
  //Get max weights
  vector<double> weights;
  double maxWeight = numeric_limits<double>::min();
  
  for (int i = 0; i<num_particles; i++){
    weights.push_back(particles[i].weight);
    
    if ( particles[i].weight > maxWeight ) {
      maxWeight = particles[i].weight;
    }
  }
  
  // Creating distributions.
  uniform_real_distribution<double> unirealdist(0.0, maxWeight);
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  
  
  // Generating index.
  int index = uniintdist(gen);

  double beta = 0.0;
  
  // the wheel
  vector<Particle> resampled_Particles;
  for(int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_Particles.push_back(particles[index]);
  }

  particles = resampled_Particles;
  

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}