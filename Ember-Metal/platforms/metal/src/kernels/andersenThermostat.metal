/**
 * Apply the Andersen thermostat to adjust particle velocities.
 */

KERNEL void applyAndersenThermostat(int numAtoms, float collisionFrequency, float kT, GLOBAL mixed4* velm, real stepSize, GLOBAL const float4* RESTRICT random,
        unsigned int randomIndex, GLOBAL const int* RESTRICT atomGroups,
        MM_THREAD_ARGS) {
    float collisionProbability = (float) (1-EXP(-collisionFrequency*stepSize));
    float x = collisionProbability/SQRT(2.0f);
    float t = RECIP(1.0f+0.3275911f*x);
    float erfcX = (0.254829592f+(-0.284496736f+(1.421413741f+(-1.453152027f+1.061405429f*t)*t)*t)*t)*t*EXP(-x*x);
    float randomRange = (float) (1.0f-erfcX);
    for (int index = GLOBAL_ID; index < numAtoms; index += GLOBAL_SIZE) {
        mixed4 velocity = velm[index];
        float4 selectRand = random[randomIndex+atomGroups[index]];
        float4 velRand = random[randomIndex+index];
        real scale = (selectRand.w > -randomRange && selectRand.w < randomRange ? 0 : 1);
        real add = (1-scale)*SQRT(kT*velocity.w);
        velocity.x = scale*velocity.x + add*velRand.x;
        velocity.y = scale*velocity.y + add*velRand.y;
        velocity.z = scale*velocity.z + add*velRand.z;
        velm[index] = velocity;
    }
}
