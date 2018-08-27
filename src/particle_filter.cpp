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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	num_particles = 100;
	default_random_engine gen;
	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.weight = 1.0;
		// Create normal distributions for x, y and theta.
		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_theta(theta, std[2]);
		// Add random Gaussian noise to each particle.
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particles.push_back(particle);
	}
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	default_random_engine gen;
	// Create 0-mean normal distributions for x, y and theta Gaussian noises.
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	for (auto& particle: particles) {
		// Check for division by 0
		if (yaw_rate != 0) {
			// Add measurements
			particle.x += (velocity / yaw_rate)
									* (sin(particle.theta + yaw_rate * delta_t)
											- sin(particle.theta));
			particle.y += (velocity / yaw_rate)
									* (cos(particle.theta)
											- cos(particle.theta + yaw_rate * delta_t));
			particle.theta += yaw_rate * delta_t;
		}
		else {
			// If yaw is not changing, particle will move in the same direction
			particle.x += velocity * delta_t * cos(particle.theta);
			particle.y += velocity * delta_t * sin(particle.theta);
		}
		// Add random Gaussian noise to each particle.
		particle.x += dist_x(gen);
		particle.y += dist_y(gen);
		particle.theta += dist_theta(gen);
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
	// Useful constants to be used in weight (probability) calculation later.
	const double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	const double exp_denom_term1 = 2 * std_landmark[0] * std_landmark[0];
  const double exp_denom_term2 = 2 * std_landmark[1] * std_landmark[1];
	// Empty previous weight vector.
	weights.clear();
	// sum of weights to be used in weight normalization later.
	double sum_weights = 0;
	// Calculate weight for each particle.
	for (Particle particle: particles) {
		// Initialize probability.
		double weight = 1.0;
		// Calculate probability for each observation.
		for (auto obs: observations) {
			// Transform each observation into map coordinates.
			LandmarkObs t_obs;
			t_obs.x = particle.x + obs.x * cos(particle.theta) - obs.y * sin(particle.theta);
			t_obs.y = particle.y + obs.x * sin(particle.theta) + obs.y * cos(particle.theta);

			// Initialize nearest landmark (minimum) distance to a high value.
			double distance_to_nearest_landmark = 10000000.0;
			// Coordinates of the map landmark to be associated with this observation.
			// These will be the mean values of the Multivariate Gaussian distribution.
			double mu_x, mu_y = 0.0;
			// Find distances to map landmarks that are in sensor's range.
			for (auto landmark: map_landmarks.landmark_list) {
				// Consider this landmark for association only if
				// predicted measurement distance in the sensor's range.
				if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {
					// Find observation's distance from landmark
					double obs_distance = dist(t_obs.x, t_obs.y, landmark.x_f, landmark.y_f);
					if (obs_distance < distance_to_nearest_landmark) {
						// New minimum distance found.
						distance_to_nearest_landmark = obs_distance;
						// Associate this observation with the nearest map landmark.
						t_obs.id = landmark.id_i;
						// Update mean values
						mu_x = landmark.x_f;
						mu_y = landmark.y_f;
					}
				}
			}
			// Calculate probability for weight using Multivariate Gaussian
			double x = t_obs.x - mu_x;
			double y = t_obs.y - mu_y;
			double prob = gauss_norm
									* exp(-(x * x / exp_denom_term1 + y * y / exp_denom_term2));
			if (prob == 0) {
				// Avoid division by 0 in normalization
				prob = 0.000001;
			}
			// Update final weight
			weight *= prob;
		}
		weights.push_back(weight);
		sum_weights += weight;
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
	// Update particles vector
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
		return particle;
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
