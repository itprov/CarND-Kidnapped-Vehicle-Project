/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::add_noise(Particle& particle, double std[]) {
	default_random_engine gen;
	// Create normal distributions for x, y and theta.
	normal_distribution<double> dist_x(particle.x, std[0]);
	normal_distribution<double> dist_y(particle.y, std[1]);
	normal_distribution<double> dist_theta(particle.theta, std[2]);
	// Add random Gaussian noise to each particle.
	particle.x = dist_x(gen);
	particle.y = dist_y(gen);
	particle.theta = dist_theta(gen);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	num_particles = 3;
	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = x;
		particle.y = y;
		particle.theta = theta;
		particle.weight = 1.0;
		// Add random Gaussian noise to each particle.
		add_noise(particle, std);
		particles.push_back(particle);
		cout << "Inited " << i << ": " << particle.x << " " << particle.y << " " << particle.theta << endl;
	}
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	default_random_engine gen;
	int i = 0;
	for (auto& particle: particles) {
		// Check for division by 0
		if (yaw_rate > 0.0001f) {
			particle.x += (velocity / yaw_rate)
									* (sin(particle.theta + yaw_rate * delta_t)
											- sin(particle.theta));
			particle.y += (velocity / yaw_rate)
									* (cos(particle.theta)
											- cos(particle.theta + yaw_rate * delta_t));
		}
		else {
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		}
		particle.theta += yaw_rate * delta_t;
		// Add random Gaussian noise to each particle.
		add_noise(particle, std_pos);
		cout << "Predicted " << i++ << ": " << particle.x << " " << particle.y << " " << particle.theta << endl;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (auto& obs_lm: observations) {
		double min_dist = 1000.0;
		for (LandmarkObs pred_lm: predicted) {
			double pred_dist = dist(pred_lm.x, pred_lm.y, obs_lm.x, obs_lm.y);
			// Update landmark observation id if it's closer to this prediction
			if (pred_dist < min_dist) {
				// New minimum distance found.
				min_dist = pred_dist;
				obs_lm.id = pred_lm.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a multi-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	double sum_weights = 0;
	weights.clear();
	for (Particle particle: particles) {
		// Transform observations into map coordinates.
		vector<LandmarkObs> transformed_observations;
		for (auto& obs : observations) {
			LandmarkObs t_obs;
			t_obs.x = particle.x + obs.x * cos(particle.theta) - obs.y * sin(particle.theta);
			t_obs.y = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta);
			transformed_observations.push_back(t_obs);
		}
		// Predict measurements to map landmarks.
		vector<LandmarkObs> predicted;
		for (auto map_landmark: map_landmarks.landmark_list) {
			LandmarkObs pred_landmark;
			pred_landmark.id = map_landmark.id_i;
			pred_landmark.x = particle.x - map_landmark.x_f;
			pred_landmark.y = particle.y - map_landmark.y_f;
			predicted.push_back(pred_landmark);
		}
		// Associate each transformed observation with a landmark id.
		dataAssociation(predicted, transformed_observations);
		// Calculate weights.
		double prob = 1.0;
		for (auto obs: transformed_observations) {
			// Find the associated landmark
			float mu_x, mu_y = 0.0f;
			for (auto landmark: map_landmarks.landmark_list) {
				if (landmark.id_i == obs.id) {
					mu_x = landmark.x_f;
					mu_y = landmark.y_f;
					break;
				}
			}
			prob *= (1 / (2 * M_PI * std_landmark[0] * std_landmark[1]))
						* exp(-
								(pow((obs.x - mu_x) / std_landmark[0], 2) / 2
								+ pow((obs.y - mu_y) / std_landmark[1], 2) / 2));
		}
		weights.push_back(prob);
		sum_weights += prob;
	}
	// Normalize weights.
	int i = 0;
	for (auto& particle: particles) {
		weights[i] /= sum_weights;
		particle.weight = weights[i++];
	}
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<int> dist_resample(weights.begin(), weights.end());
	vector<Particle> resampled_particles;
	for (int i = 0; i < num_particles; i++) {
		resampled_particles.push_back(particles[dist_resample(gen)]);
	}
	particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
