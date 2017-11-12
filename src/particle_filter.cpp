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

double distance2d(double x1, double y1, double x2, double y2) {

    return sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));

}

void ParticleFilter::init(double x, double y, double theta, double std[]) {

    num_particles = 100;

    std::default_random_engine generator;
    std::normal_distribution<double> distributionX(x, std[0]);
    std::normal_distribution<double> distributionY(y, std[1]);
    std::normal_distribution<double> distributionTheta(theta, std[2]);

    for (int i=0; i<num_particles; i++) {

        Particle p;
        p.id = i;
        p.x = distributionX(generator);
        p.y = distributionY(generator);
        p.theta = distributionTheta(generator);
        p.weight = 1.0;

        particles.push_back(p);
    }

    weights = vector<double>(num_particles, 1.0);

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

    static std::default_random_engine generator;
    static std::normal_distribution<double> distributionX(0, std_pos[0]);
    static std::normal_distribution<double> distributionY(0, std_pos[1]);
    static std::normal_distribution<double> distributionTheta(0, std_pos[2]);

    for (int nP=0; nP < num_particles; nP++) {

        Particle & p = particles[nP];

        if (abs(yaw_rate) > 1.0e-8) {
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
            p.y += velocity / yaw_rate * (-cos(p.theta + yaw_rate*delta_t) + cos(p.theta));
            p.theta += yaw_rate * delta_t;
        }
        else {
            p.x += velocity*delta_t*cos(p.theta);
            p.y += velocity*delta_t*sin(p.theta);
        }

        //add noise
        p.x += distributionX(generator);
        p.y += distributionY(generator);
        p.theta += distributionTheta(generator);

        //cout<<p.x<<" "<<p.y<<endl;
    }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs> & predicted, std::vector<LandmarkObs>& observations) {

    for (int i=0; i<observations.size(); i++) {

        double dist = distance2d(predicted[0].x, predicted[0].y, observations[i].x, observations[i].y);
        int idx = predicted[0].id;

        for (int j=1; j<predicted.size(); j++) {

            double new_dist = distance2d(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);

            if (new_dist < dist) {
                dist = new_dist;
                idx = predicted[j].id;
            }

        }

        observations[i].id = idx;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    static vector<LandmarkObs> predicted;
    predicted.clear();
    static vector<LandmarkObs> observations_map;

    double sig_x= std_landmark[0];
    double sig_y= std_landmark[1];

    for (int nP=0; nP < num_particles; nP++) {

        Particle & p = particles[nP];

        observations_map.clear();

        for (int i=0; i<observations.size(); i++) {

            double xm = p.x + observations[i].x*cos(p.theta) - observations[i].y*sin(p.theta);
            double ym = p.y + observations[i].y*cos(p.theta) + observations[i].x*sin(p.theta);

            LandmarkObs obj;
            obj.x = xm;
            obj.y = ym;
            obj.id = i;

            observations_map.push_back(obj);
        }

        predicted.clear();
        for (int i=0; i<map_landmarks.landmark_list.size(); i++) {
            LandmarkObs obj;
            obj.x = map_landmarks.landmark_list[i].x_f;
            obj.y = map_landmarks.landmark_list[i].y_f;
            obj.id = predicted.size();

            if (distance2d(obj.x, obj.y, p.x, p.y) < sensor_range) {
                predicted.push_back(obj);
            }
        }

        dataAssociation(predicted, observations_map);

        double weight = 1.0;

        for (int i=0; i<observations_map.size(); i++) {

            double x_obs = observations_map[i].x;
            double y_obs = observations_map[i].y;
            double mu_x = predicted[observations_map[i].id].x;
            double mu_y = predicted[observations_map[i].id].y;

            double gauss_norm = (1/(2 * PI * sig_x * sig_y));

            double exponent = ((x_obs - mu_x)*(x_obs - mu_x))/(2 * sig_x*sig_x);
            exponent += ((y_obs - mu_y)*(y_obs - mu_y))/(2 * sig_y*sig_y);

            double prob = gauss_norm * exp(-exponent);

            weight *= prob;
        }

        p.weight = weight;

        weights[nP] = weight;

        //cout<<nP<<" "<<weight<<endl;

    }


}

void ParticleFilter::resample() {

    static vector<double> weight_sum;

    weight_sum.clear();

    //cout<<"resample "<<num_particles<<endl;

    for (int i=0; i<num_particles; i++) {
        weight_sum.push_back(weights[i]);
        if (i > 0)
            weight_sum[i] += weight_sum[i-1];
    }

    double m = weight_sum[num_particles-1];

    if (m > 1.0e-120) {
        for (int i=0; i<num_particles; i++) {
            weight_sum[i] /= weight_sum[num_particles-1];
            //cout<<weight_sum[i]<<" ";
        }
    }
    else {
        for (int i=0; i<num_particles; i++) {
            weight_sum[i] = 1.0;
            //cout<<weight_sum[i]<<" ";
        }
    }

    //cout<<endl;

    static vector<Particle> newParticles;
    newParticles.clear();
    static vector<double> newWeights;
    newWeights.clear();

    for (int i=0; i<num_particles; i++) {
        double r = ((double) rand() / (RAND_MAX));

        int idx = min((int)(upper_bound(weight_sum.begin(), weight_sum.end(), r) - weight_sum.begin()), num_particles-1);

        newParticles.push_back(particles[idx]);
        newWeights.push_back(weights[idx]);

        //cout<<r<<" "<<idx<<" ";
    }

    //cout<<endl;

    particles = newParticles;
    weights = newWeights;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

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
