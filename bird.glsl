#version 330

in vec2 fragTexCoord;
in vec4 fragColor;

out vec4 finalColor;

uniform float time;
uniform float flap_phase;
uniform vec2 resolution;

const float BODY_HEIGHT = 0.4515625;
const float BODY_BOTTOM_RADIUS = 0.17159375;
const float BODY_TOP_RADIUS = 0.0903125;
const float BODY_TUCK_SCALE = 1.02;

const float WING_HEIGHT = 0.289;
const float WING_BOTTOM_RADIUS = 0.13005;
const float WING_TOP_RADIUS = 0.06375;
const float WING_X = 0.0903125;
const float WING_Y = 0.10115;
const float WING_BASE_ANGLE = 0.7854;

const float FLAP_SPEED = 1.3;
const float OUTWARD_FLAP_PORTION = 0.65;
const float INWARD_FLAP_PORTION = 1.0 - OUTWARD_FLAP_PORTION;
const float INWARD_FLAP_ANGLE = -0.85;
const float OUTWARD_FLAP_ANGLE = 0.55;

const vec2 BIRD_LOCAL_CENTER = vec2(0.0, 0.180625);
const float BIRD_ROTATION = -1.57079632679;
const float BIRD_SMOOTHING = 0.0289;
const vec3 BIRD_COLOR = vec3(65.0, 100.0, 200.0) / 255.0;

float sdUnevenCapsule(vec2 p, float r1, float r2, float h)
{
    p.x = abs(p.x);
    float b = (r1 - r2) / h;
    float a = sqrt(1.0 - b * b);
    float k = dot(p, vec2(-b, a));
    if (k < 0.0) return length(p) - r1;
    if (k > a * h) return length(p - vec2(0.0, h)) - r2;
    return dot(p, vec2(a, b)) - r1;
}

vec2 translate(vec2 p, vec2 offset)
{
    return p - offset;
}

vec2 rotate(vec2 p, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

vec2 scale(vec2 p, float amount)
{
    return p / amount;
}

float smoothUnion(float d1, float d2, float k)
{
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}

float flapProgress()
{
    float cycle = fract(time * FLAP_SPEED + flap_phase);
    if (cycle < OUTWARD_FLAP_PORTION) {
        return smoothstep(0.0, 1.0, cycle / OUTWARD_FLAP_PORTION);
    }
    return 1.0 - smoothstep(0.0, 1.0, (cycle - OUTWARD_FLAP_PORTION) / INWARD_FLAP_PORTION);
}

float tuck()
{
    return 1.0 - flapProgress();
}

float flapAngle()
{
    return mix(INWARD_FLAP_ANGLE, OUTWARD_FLAP_ANGLE, flapProgress());
}

float body(vec2 p)
{
    float amount = mix(1.0, BODY_TUCK_SCALE, tuck());
    return sdUnevenCapsule(scale(p, amount), BODY_BOTTOM_RADIUS, BODY_TOP_RADIUS, BODY_HEIGHT) * amount;
}

float wing(vec2 p, float side)
{
    vec2 position = vec2(side * WING_X, WING_Y);
    float angle = side * (WING_BASE_ANGLE + flapAngle());
    return sdUnevenCapsule(rotate(p - position, angle), WING_BOTTOM_RADIUS, WING_TOP_RADIUS, WING_HEIGHT);
}

float wings(vec2 p)
{
    return smoothUnion(wing(p, -1.0), wing(p, 1.0), BIRD_SMOOTHING);
}

float bird(vec2 p)
{
    return smoothUnion(body(p), wings(p), BIRD_SMOOTHING);
}

float fill(float d)
{
    float aa = fwidth(d);
    return 1.0 - smoothstep(0.0, aa, d);
}

void main()
{
    vec2 p = fragTexCoord;
    p = translate(p, vec2(0.5));
    p = rotate(p, BIRD_ROTATION);
    p = translate(p, -BIRD_LOCAL_CENTER);

    float d = bird(p);
    float mask = fill(d);

    finalColor = vec4(fragColor.rgb, mask);
}
