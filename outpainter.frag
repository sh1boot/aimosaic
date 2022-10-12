#version 330
#define MAX_WARP_ITERATIONS 4
#define MAX_PATCH_ITERATIONS 24
#define toggle(n) 0 texelFetch(iChannel3, ivec2(48+n,2),0).x
#define WARP_HIGHLIGHT vec4(0.0) // (toggle(1) * 0.5 * vec4(0.0, 0.0, 1.0, 1.0))
#define PATCH_HIGHLIGHT vec4(0.0) // (toggle(2) * 0.5 * vec4(1.0, 0.0, 1.0, 1.0))
#define OOB_HIGHLIGHT vec4(0.0) // (toggle(3) * 0.5 * vec4(1.0, 1.0, 0.0, 1.0))
#define CUT_HIGHLIGHT vec4(0.0) // (toggle(0) * 0.5 * vec4(0.0, 1.0, 0.0, 1.0))

#define iTime random.x

in vec2 f_uv;
out vec4 output_colour;

uniform vec4 random;
uniform vec2 alphapinch;
uniform vec2 roipinch;
uniform float extra_noise;
uniform sampler2D image_texture;
uniform sampler2D alpha_texture;

/////////////

mat2x2 rotmat(float a)
{
    float c = cos(a), s = sin(a);
    return mat2x2(c, s, -s, c);
}

float hash(vec2 uv) {
    const vec2 swiz = vec2(12.9898, 78.233);
    return fract(sin(dot(uv, swiz)) * 43758.5453);
}

float hash(float fx, float fy) {
    return hash(vec2(fx, fy));
}
float hash(float f) {
    return hash(vec2(f));
}

vec2 hash2(vec2 uv) {
    return vec2(hash(uv), hash(uv.yx + 1.618033989));
}

vec2 hash2(float fx, float fy) {
    return hash2(vec2(fx, fy));
}
vec2 hash2(float f) {
    return hash2(vec2(f));
}

vec4 hash4(vec2 uv) {
    return vec4(hash(fract(2.71828 * uv.y) - random.x, uv.x), hash(fract(3.14159 * uv.x) + random.y, uv.y), hash(1.61803 * uv.yx - random.z), hash(uv.xy + random.w));
}

/////////////


float ActualCutout(vec2 uv) {
    return textureLod(alpha_texture, uv, 0.0).a;
}

float BoringCutout(vec2 uv) {
    float boring = texture(alpha_texture, uv).r;
    return smoothstep(roipinch.x, roipinch.y, boring);
}

float ErodedCutout(vec2 uv)
{
    float alpha = textureLod(alpha_texture, uv, 1.5).b;
    alpha = clamp(3.0 * alpha - 2.0, 0.0, 1.0);

    alpha *= BoringCutout(uv);

    // This is just to make certain we get 0 in untouchable space:
    alpha *= ActualCutout(uv);
    return alpha;
}

float AccretedCutout(vec2 uv)
{
    float alpha = textureLod(alpha_texture, uv, 1.5).b;
    alpha = smoothstep(0.0, 1.0, 3.0 * alpha);

    // This is just to make certain we get 0 in untouchable space:
    alpha *= ActualCutout(uv);

    return alpha;
}

vec4 GetImage(vec2 uv) {
    return textureLod(image_texture, uv, 0.0) * ErodedCutout(uv);
}

float AlphaForGradient(vec2 uv) {
    // Fun effect: low LOD here lets noise bleed through from the
    // random sampling used in the blur generation.
    return textureLod(alpha_texture, uv, 3.5).g;
}

vec2 AlphaGradient(vec2 uv, float eps) {
    vec3 o = vec3(-eps, eps, 0.0);
    float up = AlphaForGradient(uv + o.zx);
    float dn = AlphaForGradient(uv + o.zy);
    float lf = AlphaForGradient(uv + o.xz);
    float rt = AlphaForGradient(uv + o.yz);

    return vec2(rt - lf, dn - up) / eps;
}

vec4 GetBackfill(vec2 uv, float fade, vec4 highlight) {
    vec2 edges = smoothstep(0.0, 0.02, uv)
               * smoothstep(0.0, 0.02, 1.0-uv);
    float edge = edges.x * edges.y;
    vec4 colour = GetImage(uv) * edge * fade;

    if (colour.a >= 0.1) {
        colour = highlight + colour * (1.0 - highlight.a);
    }
    if (edge < 0.99) {
        colour = colour + OOB_HIGHLIGHT * (1.0 - colour.a);
    }
    return colour;
}

vec4 GetWarpBackfill(vec2 uv, float gen) {
    const float eps = 0.02;
    vec2 g = AlphaGradient(uv, eps);
    float base = AlphaForGradient(uv);
    float len = length(g);
    float alpha = pow(0.7, gen);
#if 1
    len = 0.06 * (gen + 1.0);
    //len *= base * 2.0;
    // TODO: bleed some `base` into `alpha` somehow.
 //   alpha *= (base + 3.0) / 4.0;
#else
    len += gen * 0.01;
    len *= 1.0 / 64.0;
#endif
    uv += normalize(g) * len;

    return GetBackfill(uv, alpha, WARP_HIGHLIGHT);
}

vec4 GetPatchBackfill(vec2 uv, float gen) {
    float t = gen * 6.283185307 + fract(iTime * gen * 0.001);
    float r = pow(0.975, gen);
    vec2 c = 0.5 - (0.5 - vec2(hash(gen, 1.0), hash(gen, 2.0))) * 0.7;
    uv -= c;
    uv = uv * mat2x2(cos(t), sin(t), -sin(t), cos(t)) * r;
    uv += c;

    float th = (dot(uv, vec2(3.5, -3.5)) + gen + fract(iTime * 0.3)) * 6.28;
    uv += vec2(cos(th), sin(th)) * 0.02;

    return GetBackfill(uv, 1.0 - pow(0.7, gen + 1.0), PATCH_HIGHLIGHT);
}

void main(void)
{
    vec2 uv = f_uv;

    // Time varying pixel color
    vec4 col = vec4(0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4)), 1.0);

    vec4 tex = GetImage(uv);

    for (int i = 0; i < MAX_WARP_ITERATIONS; ++i) {
        if (tex.a >= 0.99) break;
        vec4 back = GetWarpBackfill(uv, float(i) * 1.618033989);
        tex = tex + back * (1.0 - tex.a);
    }
    for (int i = 0; i < MAX_PATCH_ITERATIONS; ++i) {
        if (tex.a >= 0.99) break;
        vec4 back = GetPatchBackfill(uv, float(i) * 1.618033989);
        tex = tex + back * (1.0 - tex.a);
    }
    col = tex + col * (1.0 - tex.a);

    float alpha = ActualCutout(uv);
    vec4 cut = CUT_HIGHLIGHT * (1.0 - alpha);
    col = cut + col * (1.0 - cut.a);
#if 0
    alpha = BoringCutout(uv);
    col = mix(col, vec4(1.0, 0.0, 1.0, 1.0), alpha);
#endif

    vec4 noise = hash4(random.wz + uv.xy) * 2.0 - 1.0;
    col.rgb = col.rgb + noise.rgb * extra_noise * (1.0 - AccretedCutout(uv));

    alpha = ErodedCutout(uv);
    alpha = smoothstep(alphapinch.x, alphapinch.y, alpha);
    output_colour = vec4(col.rgb, alpha);
}
