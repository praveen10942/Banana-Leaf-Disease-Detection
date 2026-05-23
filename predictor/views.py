import os
import json
import pickle
import numpy as np
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.conf import settings
from django.db.models import Count
from django.http import JsonResponse
from .models import Prediction

MODEL_DIR  = os.path.join(settings.BASE_DIR, 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'banana_disease_model.h5')
NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.pkl')

model       = None
class_names = None

def load_model():
    global model, class_names
    if model is None:
        try:
            import tensorflow as tf
            from keras.layers import DepthwiseConv2D

            class CustomDepthwiseConv2D(DepthwiseConv2D):
                def __init__(self, **kwargs):
                    if 'groups' in kwargs:
                        del kwargs['groups']
                    super().__init__(**kwargs)

            model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
            with open(NAMES_PATH, 'rb') as f:
                class_names = pickle.load(f)
        except Exception as e:
            print(f"Model load failed: {e}")

load_model()

# ── Comprehensive Disease Information & Farmer Solutions ──────────────
DISEASE_INFO = {
    'healthy': {
        'label': 'Healthy',
        'type': 'None',
        'description': 'The banana leaf appears healthy with no visible signs of infection, disease, or pest damage. The leaf shows normal green coloration and structure.',
        'main_symptom': 'No symptoms — leaf is in good condition',
        'causes': 'N/A — The plant is healthy.',
        'severity': 'None',
        'color': 'success',
        'solutions': [],
        'prevention_tips': [
            'Maintain regular watering schedule — avoid both overwatering and drought stress.',
            'Apply balanced fertilizer (NPK 14-14-14) every 2-3 months.',
            'Keep the area around the plant clean and free of debris.',
            'Inspect leaves weekly for early signs of disease or pest activity.',
            'Ensure good air circulation by maintaining proper plant spacing (2-3 meters apart).',
        ],
        'organic_remedies': [],
        'when_to_consult': 'No consultation needed. Continue regular care and monitoring.',
    },
    'black_sigatoka': {
        'label': 'Black Sigatoka',
        'type': 'Fungal',
        'description': 'Black Sigatoka (Black Leaf Streak) is caused by the fungus Mycosphaerella fijiensis. It is the most economically important leaf disease of bananas worldwide. It causes rapid leaf death, reducing fruit quality and yield by up to 50%.',
        'main_symptom': 'Dark brown to black streaks on the underside of leaves, progressing to large necrotic patches',
        'causes': 'Spread by airborne spores (ascospores and conidia) in warm, humid conditions. Thrives in temperatures of 25-28°C with high rainfall.',
        'severity': 'High',
        'color': 'danger',
        'solutions': [
            'Apply systemic fungicides like Propiconazole (Tilt) or Difenoconazole at first sign of infection.',
            'Remove and destroy severely infected leaves immediately — do not compost them.',
            'Spray protective fungicides (Mancozeb or Chlorothalonil) on a 2-3 week cycle during wet season.',
            'Improve drainage around the plantation to reduce humidity.',
            'Use resistant banana varieties if available (e.g., FHIA-01, FHIA-18).',
        ],
        'prevention_tips': [
            'De-leaf regularly — remove the oldest leaves that show early symptoms.',
            'Maintain proper spacing between plants for good air circulation.',
            'Avoid overhead irrigation; use drip irrigation instead.',
            'Apply protective fungicide sprays before the rainy season begins.',
        ],
        'organic_remedies': [
            'Apply neem oil spray (3-5 ml per liter of water) every 10-14 days.',
            'Use Trichoderma-based bio-fungicide as a preventive measure.',
            'Spray compost tea on leaves weekly to boost natural resistance.',
            'Apply potassium bicarbonate solution (1 tablespoon per gallon of water).',
        ],
        'when_to_consult': 'Consult an agronomist immediately if more than 30% of leaves show symptoms, or if the disease is spreading rapidly despite treatment.',
    },
    'yellow_sigatoka': {
        'label': 'Yellow Sigatoka',
        'type': 'Fungal',
        'description': 'Yellow Sigatoka is caused by the fungus Mycosphaerella musicola. It is less aggressive than Black Sigatoka but still causes significant yield loss. Infected leaves develop yellow spots that turn into brown lesions with grey centers.',
        'main_symptom': 'Yellow-green spots on upper leaf surface that expand into oval yellow streaks with grey-brown centers',
        'causes': 'Spread by wind and rain splash. Favored by cool, humid conditions (20-25°C). Spores germinate on wet leaf surfaces.',
        'severity': 'Moderate',
        'color': 'warning',
        'solutions': [
            'Apply copper-based fungicide (Copper Oxychloride) at early infection stage.',
            'Spray systemic fungicide like Propiconazole during severe outbreaks.',
            'Remove heavily infected leaves and destroy them away from the plantation.',
            'Improve field drainage and reduce overhead watering.',
            'Apply mineral oil emulsion sprays to reduce spore germination.',
        ],
        'prevention_tips': [
            'Practice regular de-leafing to remove infected lower leaves.',
            'Maintain good air circulation with proper plant spacing.',
            'Avoid working in the field when leaves are wet to prevent spreading spores.',
            'Monitor weekly during the rainy season for early detection.',
        ],
        'organic_remedies': [
            'Spray neem oil solution (5 ml per liter of water) every 14 days.',
            'Apply Bordeaux mixture (1% concentration) as a protective spray.',
            'Use Trichoderma harzianum as a bio-control agent in the soil.',
            'Foliar spray with garlic extract (crush 50g garlic in 1L water, filter, dilute 1:10).',
        ],
        'when_to_consult': 'Seek expert advice if the disease persists after 2-3 rounds of treatment, or if yield drops noticeably.',
    },
    'cordana': {
        'label': 'Cordana Leaf Spot',
        'type': 'Fungal',
        'description': 'Cordana Leaf Spot is caused by the fungus Cordana musae. It produces oval-shaped brown spots with yellow halos on the leaf surface. It typically attacks weakened or stressed plants and is often secondary to other diseases.',
        'main_symptom': 'Oval brown spots with distinct yellow halos, often on leaf edges and tips',
        'causes': 'Attacks plants already weakened by other diseases, nutrient deficiency, or environmental stress. Spread through wind and rain splash.',
        'severity': 'Moderate',
        'color': 'warning',
        'solutions': [
            'Apply fungicide spray (Mancozeb or Chlorothalonil) at recommended doses.',
            'Remove and destroy infected leaves — do not leave them on the ground.',
            'Strengthen plant health with balanced fertilization (especially potassium).',
            'Avoid overhead irrigation to reduce leaf wetness duration.',
            'Address any underlying stress factors (poor drainage, nutrient deficiency).',
        ],
        'prevention_tips': [
            'Keep plants well-nourished with regular fertilizer application.',
            'Improve drainage to prevent waterlogging stress.',
            'Remove old dead leaves and plant debris from around the base.',
            'Maintain proper spacing for adequate air circulation.',
        ],
        'organic_remedies': [
            'Spray neem oil (3 ml per liter of water) every 10-14 days.',
            'Apply compost tea foliar spray to boost leaf health.',
            'Use potassium-rich organic fertilizer (banana peels, wood ash).',
            'Spray diluted baking soda solution (1 tsp per liter) on affected leaves.',
        ],
        'when_to_consult': 'Consult an expert if the disease spreads to more than 40% of the plant or if combined with other infections.',
    },
    'pestalotiopsis': {
        'label': 'Pestalotiopsis',
        'type': 'Fungal',
        'description': 'Pestalotiopsis is caused by Pestalotiopsis species fungi. It causes dark necrotic spots along leaf edges that can merge into large dead areas. It often appears after mechanical damage, insect feeding, or during periods of plant stress.',
        'main_symptom': 'Dark necrotic lesions on leaf margins that expand inward, with concentric ring patterns',
        'causes': 'Enters through wounds from insects, wind damage, or pruning cuts. Thrives in humid conditions and on stressed plants.',
        'severity': 'High',
        'color': 'danger',
        'solutions': [
            'Remove infected leaves immediately and destroy them by burning.',
            'Apply systemic fungicide (Carbendazim or Thiophanate-methyl) every 14 days.',
            'Treat any insect infestations that create entry wounds for the fungus.',
            'Improve air circulation around plants by proper pruning.',
            'Apply potassium and phosphorus-rich fertilizer to boost plant immunity.',
        ],
        'prevention_tips': [
            'Minimize mechanical damage during farming operations.',
            'Control insect pests that cause feeding wounds on leaves.',
            'Ensure proper nutrition — stressed plants are more susceptible.',
            'Sterilize pruning tools between plants using 70% alcohol.',
        ],
        'organic_remedies': [
            'Apply neem oil spray (5 ml per liter) fortnightly.',
            'Use Trichoderma viride as a bio-control agent in the root zone.',
            'Spray copper-based organic fungicide (Bordeaux mixture at 1%).',
            'Strengthen plants with compost and vermicompost application.',
        ],
        'when_to_consult': 'Seek professional help if the disease is rapidly spreading or if combined with other fungal infections.',
    },
    'moko': {
        'label': 'Moko Disease',
        'type': 'Bacterial',
        'description': 'Moko Disease is caused by the bacterium Ralstonia solanacearum. It is one of the most destructive bacterial diseases of bananas. The bacteria invade the vascular system, blocking water transport and causing rapid wilting and plant death.',
        'main_symptom': 'Young leaves turn yellow and wilt, followed by browning and collapse. Cutting the pseudostem reveals dark brown vascular discoloration.',
        'causes': 'Spread through contaminated soil, tools, and infected planting material. Insects (especially bees visiting male flowers) can also transmit it.',
        'severity': 'Critical',
        'color': 'danger',
        'solutions': [
            '⚠️ There is NO chemical cure for Moko Disease. Infected plants MUST be destroyed.',
            'Immediately uproot and destroy (burn) the entire infected plant including the corm.',
            'Remove all banana plants within a 5-meter radius of the infected plant as a quarantine measure.',
            'Disinfect all tools used on infected plants with 10% bleach or formalin solution.',
            'Do not replant bananas in the infected area for at least 12 months.',
            'Apply lime to the infected soil area to suppress bacterial survival.',
        ],
        'prevention_tips': [
            'Use only certified disease-free planting material (suckers or tissue culture plants).',
            'Remove male flower buds (de-budding) to prevent insect-mediated spread.',
            'Disinfect cutting tools between each plant with bleach or alcohol.',
            'Avoid planting bananas near Heliconia or other Musa species that can harbor the bacteria.',
        ],
        'organic_remedies': [
            'There are no effective organic cures — prevention is the only option.',
            'Apply bio-fumigation using Brassica cover crops before replanting.',
            'Use mycorrhizal inoculants when replanting to boost root defense.',
            'Practice crop rotation with non-host crops (rice, corn) for 2+ years.',
        ],
        'when_to_consult': '🚨 IMMEDIATELY consult your local agricultural extension office. Moko Disease is a notifiable disease in many countries and requires quarantine measures.',
    },
    'panama': {
        'label': 'Panama Disease (Fusarium Wilt)',
        'type': 'Fungal',
        'description': 'Panama Disease is caused by the soil-borne fungus Fusarium oxysporum f. sp. cubense. It is one of the most devastating diseases of bananas and was responsible for the near-extinction of the Gros Michel banana variety. The fungus blocks the plant\'s vascular system.',
        'main_symptom': 'Older leaves turn yellow starting from the edges, then wilt and hang down like a skirt around the pseudostem. Splitting of the pseudostem base.',
        'causes': 'Soil-borne fungus that can persist in soil for 30+ years. Spread through contaminated soil, water runoff, and infected planting material.',
        'severity': 'Critical',
        'color': 'danger',
        'solutions': [
            '⚠️ There is NO effective fungicide treatment for Panama Disease.',
            'Remove and destroy the entire infected plant including roots and surrounding soil.',
            'DO NOT replant bananas in the infected area — the fungus persists in soil for decades.',
            'Establish a quarantine zone around the infection site.',
            'Use resistant varieties (Cavendish for Race 1, GCTCV-218 for TR4).',
            'Report to agricultural authorities if Tropical Race 4 (TR4) is suspected.',
        ],
        'prevention_tips': [
            'Use certified disease-free tissue culture plantlets for new plantations.',
            'Implement strict biosecurity — clean shoes and equipment before entering fields.',
            'Maintain soil health with organic matter and proper pH (6.0-7.0).',
            'Avoid moving soil or plant material between farms.',
        ],
        'organic_remedies': [
            'Apply beneficial microorganisms (Trichoderma, Bacillus subtilis) to soil.',
            'Bio-fumigation with Brassica crops before replanting in nearby clean soil.',
            'Heavy application of well-composted organic matter to improve soil biology.',
            'Mycorrhizal fungi inoculation can provide some suppressive effect.',
        ],
        'when_to_consult': '🚨 IMMEDIATELY contact your district agricultural office. Panama Disease (especially TR4) is a quarantine pest that requires government-level intervention.',
    },
    'insect_pest': {
        'label': 'Insect Pest Damage',
        'type': 'Pest',
        'description': 'Banana plants can be attacked by various insect pests including banana weevil borers, thrips, aphids, and leaf-rolling caterpillars. Insect damage weakens the plant and creates entry points for fungal and bacterial diseases.',
        'main_symptom': 'Holes, tunnels, rolled leaves, chewed edges, silvery streaks, or sticky honeydew on leaf surfaces',
        'causes': 'Various insects: banana weevils (Cosmopolites sordidus), thrips, aphids, leaf miners, and caterpillars. Population increases during warm, dry conditions.',
        'severity': 'Moderate',
        'color': 'warning',
        'solutions': [
            'Identify the specific pest before treatment — different insects need different approaches.',
            'For banana weevils: apply Chlorpyrifos granules at the base of the plant.',
            'For thrips/aphids: spray Imidacloprid or Thiamethoxam-based insecticide.',
            'For caterpillars: apply Bacillus thuringiensis (Bt) spray — effective and eco-friendly.',
            'For severe infestations: use systemic insecticide (Acetamiprid) as a drench.',
            'Remove and destroy heavily infested plant parts.',
        ],
        'prevention_tips': [
            'Inspect plants regularly (weekly) for early signs of pest activity.',
            'Maintain field hygiene — remove old leaf sheaths and plant debris.',
            'Use pheromone traps for banana weevil monitoring and mass trapping.',
            'Encourage natural predators (spiders, ladybugs, parasitic wasps).',
        ],
        'organic_remedies': [
            'Spray neem oil solution (5 ml per liter) every 7-10 days for soft-bodied pests.',
            'Use Beauveria bassiana bio-insecticide for weevil control.',
            'Apply diatomaceous earth around plant bases to deter crawling insects.',
            'Garlic-chili spray: blend 50g garlic + 10 chilies in 1L water, filter, dilute 1:5, spray.',
            'Release Trichogramma parasitic wasps for caterpillar biocontrol.',
        ],
        'when_to_consult': 'Consult an entomologist if you cannot identify the pest, or if infestation persists despite treatment.',
    },
    'bract_mosaic_virus': {
        'label': 'Bract Mosaic Virus',
        'type': 'Viral',
        'description': 'Banana Bract Mosaic Virus (BBrMV) is a potyvirus that causes mosaic patterns on bracts, leaves, and pseudostems. It reduces plant vigor, delays flowering, and can reduce bunch weight by 40-70%. It is transmitted by aphid vectors.',
        'main_symptom': 'Spindle-shaped reddish-brown streaks on pseudostem, mosaic patterns on bracts and leaves, and chlorotic/necrotic streaks',
        'causes': 'Transmitted by aphids (especially Pentalonia nigronervosa and Aphis gossypii). Also spread through infected planting material (suckers).',
        'severity': 'High',
        'color': 'danger',
        'solutions': [
            '⚠️ There is NO cure for viral diseases. Focus on containment and prevention.',
            'Immediately rogue out (remove and destroy) infected plants to prevent spread.',
            'Control aphid vectors aggressively with insecticide sprays (Imidacloprid).',
            'Replace with virus-free tissue culture plantlets from certified nurseries.',
            'Destroy all removed plant material by burning — do not compost.',
            'Monitor surrounding plants closely for 3-6 months after removal.',
        ],
        'prevention_tips': [
            'Source planting material ONLY from certified, virus-indexed nurseries.',
            'Control aphid populations with regular insecticide application or reflective mulches.',
            'Inspect new planting material carefully before planting.',
            'Maintain good field sanitation to reduce aphid breeding sites.',
        ],
        'organic_remedies': [
            'No organic cure exists for the virus itself.',
            'Control aphid vectors using neem oil spray (5 ml/L) every 7 days.',
            'Use yellow sticky traps to monitor and reduce aphid populations.',
            'Plant barrier crops around the field to reduce aphid migration.',
            'Apply reflective mulch (silver-colored) to repel aphids.',
        ],
        'when_to_consult': '🚨 Consult a plant virologist or your agricultural extension office. Virus diseases spread quickly through insect vectors.',
    },
}


def preprocess_image(image_path):
    import tensorflow as tf
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def landing(request):
    if request.user.is_authenticated:
        return render(request, 'predictor/landing.html')
    return render(request, 'predictor/landing.html')


def upload(request):
    if request.method == 'POST':
        is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.POST.get('ajax') == 'true'

        if 'image' not in request.FILES:
            if is_ajax:
                return JsonResponse({'success': False, 'message': 'Please select an image.'})
            messages.error(request, 'Please select an image.')
            return redirect('upload')

        img_file = request.FILES['image']

        if img_file.content_type not in ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']:
            if is_ajax:
                return JsonResponse({'success': False, 'message': 'Only JPG, PNG or WEBP images are supported.'})
            messages.error(request, 'Only JPG, PNG or WEBP images are supported.')
            return redirect('upload')

        if model is None:
            if is_ajax:
                return JsonResponse({'success': False, 'message': 'Diagnostic engine is offline. Model is not loaded (run train_cnn.py first).'})
            messages.error(request, 'Model is not loaded. Please run train_cnn.py first.')
            return redirect('upload')

        if request.user.is_authenticated:
            pred_obj = Prediction(user=request.user, image=img_file, disease='Pending', confidence=0.0)
            pred_obj.save()
            img_path  = os.path.join(settings.MEDIA_ROOT, pred_obj.image.name)
            image_url = pred_obj.image.url
        else:
            from django.core.files.storage import default_storage
            from django.core.files.base import ContentFile
            import uuid
            ext = os.path.splitext(img_file.name)[1] or '.jpg'
            guest_filename = f"uploads/guest_{uuid.uuid4().hex}{ext}"
            saved_path = default_storage.save(guest_filename, ContentFile(img_file.read()))
            img_path = os.path.join(settings.MEDIA_ROOT, saved_path)
            image_url = default_storage.url(saved_path)

            class GuestPrediction:
                def __init__(self, url):
                    class MockImage:
                        def __init__(self, u):
                            self.url = u
                    self.image = MockImage(url)
                    self.disease = 'Pending'
                    self.confidence = 0.0
                    self.is_healthy = False
                    self.created_at = None
            pred_obj = GuestPrediction(image_url)

        input_arr = preprocess_image(img_path)
        preds     = model.predict(input_arr, verbose=0)[0]

        top_idx   = int(np.argmax(preds))
        top_conf  = float(preds[top_idx]) * 100
        raw_class = class_names[top_idx]

        top3_idx = np.argsort(preds)[::-1][:3]
        top3 = [
            {
                'label': DISEASE_INFO.get(class_names[i], {}).get('label', class_names[i].title()),
                'confidence': round(float(preds[i]) * 100, 1),
                'color': DISEASE_INFO.get(class_names[i], {}).get('color', 'secondary'),
            }
            for i in top3_idx
        ]

        info = DISEASE_INFO.get(raw_class, {
            'label': raw_class.title(),
            'type': 'Unknown',
            'description': 'Unknown condition detected.',
            'main_symptom': 'Unknown',
            'causes': 'Unknown',
            'severity': 'Unknown',
            'color': 'secondary',
            'solutions': ['Please consult a local agronomist for proper diagnosis.'],
            'prevention_tips': ['Regular monitoring and good agricultural practices are recommended.'],
            'organic_remedies': [],
            'when_to_consult': 'Consult an agronomist as soon as possible.',
        })

        pred_obj.disease    = info['label']
        pred_obj.confidence = round(top_conf, 1)
        pred_obj.is_healthy = (raw_class == 'healthy')
        
        if request.user.is_authenticated:
            pred_obj.save()

        if request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.POST.get('ajax') == 'true':
            return JsonResponse({
                'success': True,
                'disease': info['label'],
                'confidence': round(top_conf, 1),
                'is_healthy': pred_obj.is_healthy,
                'image_url': image_url,
                'info': info,
                'top3': top3,
            })

        return render(request, 'predictor/result.html', {
            'pred': pred_obj,
            'info': info,
            'top3': top3,
            'raw_conf': round(top_conf, 1),
        })

    return render(request, 'predictor/upload.html')


@login_required
def history(request):
    filter_type = request.GET.get('filter', 'all')
    qs = Prediction.objects.filter(user=request.user)
    if filter_type == 'healthy':
        qs = qs.filter(is_healthy=True)
    elif filter_type == 'diseased':
        qs = qs.filter(is_healthy=False)
    preds = qs[:30]
    return render(request, 'predictor/history.html', {'preds': preds, 'filter_type': filter_type})


@login_required
def dashboard(request):
    total    = Prediction.objects.filter(user=request.user).count()
    healthy  = Prediction.objects.filter(user=request.user, is_healthy=True).count()
    diseased = total - healthy
    recent   = Prediction.objects.filter(user=request.user)[:5]
    disease_counts = (
        Prediction.objects
        .filter(user=request.user, is_healthy=False)
        .values('disease')
        .annotate(count=Count('disease'))
        .order_by('-count')
    )
    return render(request, 'predictor/dashboard.html', {
        'total': total,
        'healthy': healthy,
        'diseased': diseased,
        'recent': recent,
        'disease_counts': disease_counts,
    })


def about(request):
    return render(request, 'predictor/about.html')


def model_stats(request):
    metrics_path = os.path.join(MODEL_DIR, 'metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    return render(request, 'predictor/model_stats.html', {'metrics': metrics})


def privacy_policy(request):
    return render(request, 'predictor/privacy_policy.html')


def terms_conditions(request):
    return render(request, 'predictor/terms_conditions.html')


def cookie_policy(request):
    return render(request, 'predictor/cookie_policy.html')


def disclaimer(request):
    return render(request, 'predictor/disclaimer.html')


def get_outbreaks(request):
    outbreaks = [
        {
            'city': 'Jalgaon, MH',
            'coords': [21.0077, 75.5626],
            'disease': 'Black Sigatoka',
            'severity': 'Critical',
            'cases': 14,
            'status': 'Active Outbreak',
            'color': '#ef4444'
        },
        {
            'city': 'Trichy, TN',
            'coords': [10.7905, 78.7047],
            'disease': 'Panama Wilt TR4',
            'severity': 'Critical',
            'cases': 8,
            'status': 'Quarantine Protocol',
            'color': '#ef4444'
        },
        {
            'city': 'Nanded, MH',
            'coords': [19.1383, 77.3210],
            'disease': 'Yellow Sigatoka',
            'severity': 'Moderate',
            'cases': 23,
            'status': 'Under Treatment',
            'color': '#f97316'
        },
        {
            'city': 'Hajipur, BH',
            'coords': [25.6858, 85.2215],
            'disease': 'Moko Wilt Disease',
            'severity': 'Critical',
            'cases': 5,
            'status': 'Eradication Alert',
            'color': '#ef4444'
        },
        {
            'city': 'Anand, GJ',
            'coords': [22.5645, 72.9289],
            'disease': 'Cordana Leaf Spot',
            'severity': 'Low',
            'cases': 19,
            'status': 'Stable / Contained',
            'color': '#10b981'
        },
        {
            'city': 'Bhusawal, MH',
            'coords': [21.0476, 75.7876],
            'disease': 'Insect Pest Damage',
            'severity': 'Moderate',
            'cases': 34,
            'status': 'Monitoring Active',
            'color': '#f97316'
        }
    ]
    return JsonResponse({'success': True, 'outbreaks': outbreaks})