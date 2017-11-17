#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""

import numpy as np
import argparse
import io
import nltk
import pickle
import requests
import re

from parser import depparse_ssplit

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
from os.path import join as pjoin

import json
from itertools import izip

from copy import deepcopy as cp

np.random.seed(123)

def setup_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def test():
    test_items = [
        {
            "sentence": "After release , it received downloadable content , along with an expanded edition in November of that year .",
            "previous_sentence": "It met with positive sales in Japan , and was praised by both Japanese and western critics .",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game .",
            "previous_sentence": ".",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "After the release of Valkyria Chronicles II , the staff took a look at both the popular response for the game and what they wanted to do next for the series .",
            "previous_sentence": "NA",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "Kotaku 's Richard Eisenbeis was highly positive about the game , citing is story as a return to form after Valkyria Chronicles II and its gameplay being the best in the series .",
            "previous_sentence": "NA",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "Valkyria of the Battlefield 3 : The Flower of the Nameless Oath ) , illustrated by Naoyuki Fujisawa and eventually released in two volumes after being serialized in Dengeki Maoh between 2011 and 2012 ; and Senjō no Valkyria 3 : <unk> Unmei no <unk> <unk> ( 戦場のヴァルキュリア3 <unk> , lit .",
            "previous_sentence": "They were Senjō no Valkyria 3 : Namo <unk> <unk> no Hana ( 戦場のヴァルキュリア3 <unk> , lit .",
            "marker": "after",
            "output": ('eventually released in two volumes', 'being serialized in Dengeki Maoh between 2011 and 2012')
        },
        {
            "sentence": "After taking up residence , her health began to deteriorate .",
            "previous_sentence": "She restored a maisonette in Storrington , Sussex , England , bequeathed by her friend Edith Major , and named it St. Andrew 's .",
            "marker": "after",
            "output": (', her health began to deteriorate .', 'taking up residence')
        },
        {
            "sentence": "While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers .",
            "previous_sentence": "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II .",
            "marker": "also",
            "output": ('it retained the standard features of the series', ', it underwent multiple adjustments , such as making the game more forgiving for series newcomers .')
        },
        {
            "sentence": "It was also adapted into manga and an original video animation series .",
            "previous_sentence": "After release , it received downloadable content , along with an expanded edition in November of that year .",
            "marker": "also",
            "output": ('After release , it received downloadable content , along with an expanded edition in November of that year .', 'It was adapted into manga and an original video animation series .')
        },
        {
            "sentence": "There are also love simulation elements related to the game 's two main heroines , although they take a very minor role .",
            "previous_sentence": "After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game .",
            "marker": "also",
            "output": ("After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game .", "There are love simulation elements related to the game 's two main heroines , although they take a very minor role .")
        },
        {
            "sentence": "The music was composed by Hitoshi Sakimoto , who had also worked on the previous Valkyria Chronicles games .",
            "previous_sentence": ".",
            "marker": "also",
            "output": None
        },
        {
            "sentence": "Gallian Army Squad 422 , also known as \" The Nameless \" , are a penal military unit composed of criminals , foreign deserters , and military offenders whose real names are erased from the records and thereon officially referred to by numbers .",
            "previous_sentence": "The game takes place during the Second Europan War .",
            "marker": "also",
            "output": None
        },
        {
            "sentence": "In a preview of the TGS demo , Ryan Geddes of IGN was left excited as to where the game would go after completing the demo , along with enjoying the improved visuals over Valkyria Chronicles II .",
            "previous_sentence": ".",
            "marker": "after",
            "output": ('as to where the game would go', 'completing the demo')
        },
        {
            "sentence": "The units comprising the infantry force of Van Dorn 's Army of the West were the 1st and 2nd Arkansas Mounted Rifles were also armed with M1822 flintlocks from the Little Rock Arsenal .",
            "previous_sentence": "The 9th and 10th Arkansas , four companies of Kelly 's 9th Arkansas Battalion , and the 3rd Arkansas Cavalry Regiment were issued flintlock Hall 's Rifles .",
            "marker": "also",
            "output": None
        },
        {
            "sentence": "The Tower Building of the Little Rock Arsenal , also known as U.S. Arsenal Building , is a building located in MacArthur Park in downtown Little Rock , Arkansas .",
            "previous_sentence": ".",
            "marker": "also",
            "output": None
        },
        {
            "sentence": "It has also been the headquarters of the Little Rock Æsthetic Club since 1894 .",
            "previous_sentence": "It was home to the Arkansas Museum of Natural History and Antiquities from 1942 to 1997 and the MacArthur Museum of Arkansas Military History since 2001 .",
            "marker": "also",
            "output": ('It was home to the Arkansas Museum of Natural History and Antiquities from 1942 to 1997 and the MacArthur Museum of Arkansas Military History since 2001 .', 'It has been the headquarters of the Little Rock \xc3\x86sthetic Club since 1894 .')
        },
        {
            "sentence": "It was also the starting place of the Camden Expedition .",
            "previous_sentence": "Besides being the last remaining structure of the original Little Rock Arsenal and one of the oldest buildings in central Arkansas , it was also the birthplace of General Douglas MacArthur , who became the supreme commander of US forces in the South Pacific during World War II .",
            "marker": "also",
            "output": ('Besides being the last remaining structure of the original Little Rock Arsenal and one of the oldest buildings in central Arkansas , it was also the birthplace of General Douglas MacArthur , who became the supreme commander of US forces in the South Pacific during World War II .', 'It was the starting place of the Camden Expedition .')
        },
        {
            "sentence": "Fey 's projects after 2008 include a voice role in the English @-@ language version of the Japanese animated film Ponyo .",
            "previous_sentence": ".",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "The main cave ( Cave 1 , or the Great Cave ) was a Hindu place of worship until Portuguese rule began in 1534 , after which the caves suffered severe damage .",
            "previous_sentence": ".",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "This movement , although not authorized by me , has assumed such an aspect that it becomes my duty , as the executive of this <unk> , to interpose my official authority to prevent a collision between the people of the State and the Federal troops under your command .",
            "previous_sentence": "This movement is prompted by the feeling that pervades the citizens of this State that in the present emergency the arms and munitions of war in the Arsenal should be under the control of the State authorities , in order to their security .",
            "marker": "although",
            "output": None
        },
        {
            "sentence": "Dunnington was selected to head the ordnance works at Little Rock , and although he continued to draw his pay from the Confederate Navy Department , he was placed in charge of all Confederate ordnance activities ( which included artillery functions ) there with the rank of lieutenant colonel .",
            "previous_sentence": "Ponchartrain , which had been brought to Little Rock in hopes of converting it to an ironclad .",
            "marker": "although",
            "output": (', he was placed in charge of all Confederate ordnance activities ( which included artillery functions ) there with the rank of lieutenant colonel', 'he continued to draw his pay from the Confederate Navy Department')
        },
        {
            "sentence": "The development of a national team faces challenges similar to those across Africa , although the national football association has four staff members focusing on women 's football .",
            "previous_sentence": "The Gambia has two youth teams , an under @-@ 17 side that has competed in FIFA U @-@ 17 Women 's World Cup qualifiers , and an under @-@ 19 side that withdrew from regional qualifiers for an under @-@ 19 World Cup .",
            "marker": "although",
            "output": ('The development of a national team faces challenges similar to those across Africa , .', "the national football association has four staff members focusing on women 's football")
        },
        {
            "sentence": "Although this species is discarded when caught , it is more delicate @-@ bodied than other maskrays and is thus unlikely to survive encounters with trawling gear .",
            "previous_sentence": "In the present day , this is mostly caused by Australia 's Northern Prawn Fishery , which operates throughout its range .",
            "marker": "although",
            "output": (', it is more delicate-bodied than other maskrays and is thus unlikely to survive encounters with trawling gear .', 'this species is discarded when caught')
        },
        {
            "sentence": "In the nineteenth @-@ century , the mound was higher on the western end of the tomb , although this was removed by excavation to reveal the sarsens beneath during the 1920s .",
            "previous_sentence": "The earthen mound that once covered the tomb is now visible only as an undulation approximately 1 foot , 6 inches in height .",
            "marker": "although",
            "output": ('In the nineteenth-century , the mound was higher on the western end of the tomb , .', 'this was removed by excavation to reveal the sarsens beneath during the 1920s')
        },
        {
            "sentence": "In 1880 , the archaeologist Flinders Petrie included the existence of the stones at \" <unk> \" in his list of Kentish earthworks ; although noting that a previous commentator had described the stones as being in the shape of an oval , he instead described them as forming \" a rectilinear enclosure \" around the chamber .",
            "previous_sentence": "He believed that the monument consisted of both a \" chamber \" and an \" oval \" of stones , suggesting that they were \" two distinct erections \" .",
            "marker": "although",
            "output": (', he instead described them as forming " a rectilinear enclosure " around the chamber', 'noting that a previous commentator had described the stones as being in the shape of an oval')
        },
        {
            "sentence": "She was not damaged although it took over a day to pull her free .",
            "previous_sentence": "Webb demonstrated his aggressiveness when he attempted to sortie on the first spring tide ( 30 May ) after taking command , but Atlanta 's forward engine broke down after he had passed the obstructions , and the ship ran aground .",
            "marker": "although",
            "output": ('She was not damaged .', 'it took over a day to pull her free')
        },
        {
            "sentence": "Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable .",
            "previous_sentence": "Senjō no Valkyria 3 : <unk> Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit .",
            "marker": "and",
            "output": None
        },
        {
            "sentence": "Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the \" Nameless \" , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit \" <unk> Raven \" .",
            "previous_sentence": "Released in January 2011 in Japan , it is the third game in the Valkyria series .",
            "marker": "and",
            "output": ('Employing the same fusion of tactical and real-time gameplay as its predecessors , the story runs parallel to the first game .', 'follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " <unk> Raven "')
        },
        {
            "sentence": "Character designer <unk> Honjou and composer Hitoshi Sakimoto both returned from previous entries , along with Valkyria Chronicles II director Takeshi Ozawa .",
            "previous_sentence": "While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers .",
            "marker": "and",
            "output": None
        },
        {
            "sentence": "It met with positive sales in Japan , and was praised by both Japanese and western critics .",
            "previous_sentence": ".",
            "marker": "and",
            "output": ('It met with positive sales in Japan , .', 'was praised by both Japanese and western critics')
        },
        {
            "sentence": "It was also adapted into manga and an original video animation series .",
            "previous_sentence": "After release , it received downloadable content , along with an expanded edition in November of that year .",
            "marker": "and",
            "output": None
        },
        {
            "sentence": "As with previous <unk> Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces .",
            "previous_sentence": ".",
            "marker": "and",
            ## the "where" here is not quite right, but I don't know how to resolve that the way the dependency parse works...
            "output": ('where players take control of a military unit', 'take part in missions against enemy forces')
        },
        {
            "sentence": "Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through unvoiced text .",
            "previous_sentence": "As with previous <unk> Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces .",
            "marker": "and",
            "output": None
        },
        {
            ## the "that" here is not quite right, but I don't know how to resolve that the way the dependency parse works...
            "sentence": "The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked .",
            "previous_sentence": "Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through unvoiced text .",
            "marker": "and",
            "output": ('that can be freely scanned through', 'replayed as they are unlocked')
        },
        {
            ## the "where" here is not quite right, but I don't know how to resolve that the way the dependency parse works...
            "sentence": "Outside missions , the player characters rest in a camp , where units can be customized and character growth occurs .",
            "previous_sentence": "The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player .",
            "marker": "and",
            "output": ('where units can be customized', 'character growth occurs')
        },
        {
            "sentence": "According to Sega , this was due to poor sales of Valkyria Chronicles II and the general unpopularity of the PSP in the west .",
            "previous_sentence": "Unlike its two predecessors , Valkyria Chronicles III was not released in the west .",
            "marker": "and",
            "output": None
        },
        {
            "sentence": "As with previous <unk> Chronicles games , Valkyria Chronicles III is a tactical role @-@ playing game where players take control of a military unit and take part in missions against enemy forces .",
            "previous_sentence": ".",
            "marker": "as",
            "output": None
        },
        {
            "sentence": "The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked .",
            "previous_sentence": "Stories are told through comic book @-@ like panels with animated character portraits , with characters speaking partially through voiced speech bubbles and partially through unvoiced text .",
            "marker": "as",
            "output": ('replayed', 'they are unlocked')
        },
        {
            "sentence": "As the Nameless officially do not exist , the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war .",
            "previous_sentence": ".",
            "marker": "as",
            "output": (', the upper echelons of the Gallian Army exploit the concept of plausible deniability in order to send them on missions that would otherwise make Gallia lose face in the war .', 'the Nameless officially do not exist')
        },
        {
            "sentence": "He served as the back @-@ up to Allen York during the game , and the following day , he signed a contract for the remainder of the year .",
            "previous_sentence": "After being eliminated from the NCAA Tournament just days prior , Hunwick skipped an astronomy class and drove his worn down 2003 Ford Ranger to Columbus to make the game .",
            "marker": "as",
            "output": None
        },
        {
            "sentence": "The tower was an edifice of great value for astronomical observations made using a sundial as they provided essential confirmation of the need to reform the Julian calendar .",
            "previous_sentence": "Four stages of progressive development have occurred since it was first established .",
            "marker": "as",
            "output": ('The tower was an edifice of great value for astronomical observations made using a sundial .', 'they provided essential confirmation of the need to reform the Julian calendar')
        },
        {
            "sentence": "The first stage of building of the tower , as recorded by Leo XIII in his motu proprio Ut <unk> of 1891 , is credited to Pope Gregory XIII , Pope from 1572 to 1585 .",
            "previous_sentence": ".",
            "marker": "as",
            "output": None
        },
        {
            "sentence": "Its instrumentation , apart from many normal devices ( such as meteorological and magnetic equipment , with a seismograph and a small transit and pendulum clock , ) was noted for the <unk> Telescope .",
            "previous_sentence": "The original observatory was then set up above the second level of the tower with the agreement of Pope Pius VI .",
            "marker": "as",
            "output": None
        },
        {
            "sentence": "As a result of these modifications , the original library was moved to the Pontifical Academy Lincei , and the old meteorological and seismic instruments were shifted to the Valle di Pompei observatory .",
            "previous_sentence": "A new 16 @-@ inch visual telescope , called Torre Pio X , was erected in the second tower .",
            "marker": "as",
            "output": None
        },
        {
            "sentence": "Bulloch and the passengers embarked in the steamer while Bulloch dispatched a letter to his financial agents instructing them to settle damages with the brig 's owners because he could not afford to take the time to deal with the affair lest he and Fingal be detained .",
            "previous_sentence": "On the night 14 / 15 October , as she was slowly rounding the breakwater at Holyhead , Fingal rammed and sank the Austrian brig <unk> , slowly swinging at anchor without lights .",
            "marker": "because",
            "output": ("Bulloch dispatched a letter to his financial agents instructing them to settle damages with the brig 's owners", 'he could not afford to take the time to deal with the affair lest he and Fingal be detained')
        },
        {
            ## I'm OK with this.
            "sentence": "Deceased humans were called nṯr because they were considered to be like the gods , whereas the term was rarely applied to many of Egypt 's lesser supernatural beings , which modern scholars often call \" demons \" .",
            "previous_sentence": "The term nṯr may have applied to any being that was in some way outside the sphere of everyday life .",
            "marker": "because",
            "output": ("Deceased humans were called nṯr , whereas the term was rarely applied to many of Egypt 's lesser supernatural beings , which modern scholars often call \" demons \" .", "they were considered to be like the gods")
        },
        {
            "sentence": "Because many deities in later times were strongly tied to particular towns and regions , many scholars have suggested that the pantheon formed as disparate communities coalesced into larger states , spreading and intermingling the worship of the old local deities .",
            "previous_sentence": "Predynastic Egypt originally consisted of small , independent villages .",
            "marker": "because",
            "output": (', many scholars have suggested that the pantheon formed as disparate communities coalesced into larger states , spreading and intermingling the worship of the old local deities .', 'many deities in later times were strongly tied to particular towns and regions')
        },
        {
            "sentence": "Most myths about them lack highly developed characters and plots , because the symbolic meaning of the myths was more important than elaborate storytelling .",
            "previous_sentence": "Their behavior is inconsistent , and their thoughts and motivations are rarely stated .",
            "marker": "because",
            "output": ('Most myths about them lack highly developed characters and plots , .', 'the symbolic meaning of the myths was more important than elaborate storytelling')
        },
        {
            "sentence": "Because of the gods ' multiple and overlapping roles , deities can have many epithets — with more important gods accumulating more titles — and the same epithet can apply to many deities .",
            "previous_sentence": "In addition to their names , gods were given epithets , like \" possessor of splendor \" , \" ruler of Abydos \" , or \" lord of the sky \" , that describe some aspect of their roles or their worship .",
            "marker": "because",
            "output": None
        },
        {
            "sentence": "In October 1941 , the newly installed mayor of Zagreb , Ivan Werner , issued a decree ordering the demolition of the Praška Street synagogue , ostensibly because it did not fit into the city 's master plan .",
            "previous_sentence": ".",
            "marker": "because",
            "output": ('In October 1941 , the newly installed mayor of Zagreb , Ivan Werner , issued a decree ordering the demolition of the Praška Street synagogue , .', "ostensibly it did not fit into the city 's master plan")
        },
        {
            "sentence": "Because this was Jordan 's first championship since his father 's murder , and it was won on Father 's Day , Jordan reacted very emotionally upon winning the title , including a memorable scene of him crying on the locker room floor with the game ball .",
            "previous_sentence": "He also achieved only the second sweep of the MVP Awards in the All @-@ Star Game , regular season and NBA Finals , Willis Reed having achieved the first , during the 1969 – 70 season .",
            "marker": "because",
            "output": (', Jordan reacted very emotionally upon winning the title , including a memorable scene of him crying on the locker room floor with the game ball .', "this was Jordan 's first championship since his father 's murder , and it was won on Father 's Day")
        },
        {
            "sentence": "Archaeologists have been unable to prove whether this adoption of farming was because of a new influx of migrants coming in from continental Europe or because the indigenous Mesolithic Britons came to adopt the agricultural practices of continental societies .",
            "previous_sentence": "Beginning in the fifth millennium BCE , it saw a widespread change in lifestyle as the communities living in the British Isles adopted agriculture as their primary form of subsistence , abandoning the hunter @-@ gatherer lifestyle that had characterised the preceding Mesolithic period .",
            "marker": "because",
            "output": None
        },
        {
            "sentence": "Her sister Dorothy taught kindergarten in two private schools before opening a kindergarten at home .",
            "previous_sentence": "Following her father ’ s death in June 1912 , the seventeen @-@ year @-@ old Barker submitted art and poetry to My Magazine , Child ’ s Own , Leading Strings , and Raphael Tuck annuals in an effort to support both her mother and sister .",
            "marker": "before",
            "output": ('Her sister Dorothy taught kindergarten in two private schools .', 'opening a kindergarten at home')
        },
        {
            "sentence": "Most of the body lacks dermal denticles ; a midline row of 4 – 13 small , closely spaced thorns is present behind the spiracles , and another row of 0 – 4 thorns before the stings .",
            "previous_sentence": "After the stings , the tail becomes slender and bears a long ventral fin fold and a much shorter , lower dorsal fin fold .",
            "marker": "before",
            "output": None
        },
        {
            ## bad sentence tokenization here...
            "sentence": "@ 8 seconds allowing the Kings time to score the tying goal , before winning in overtime .",
            "previous_sentence": "The team was involved in a controversial loss to the Los Angeles Kings , when the Staples Center clock appeared to freeze at 1 @.",
            "marker": "before",
            "output": ('@ 8 seconds allowing the Kings time to score the tying goal , .', 'winning in overtime')
        },
        {
            "sentence": "Sanford started 12 consecutive games before Steve Mason made his next start .",
            "previous_sentence": "@ 38 goals against average and <unk> save percentage over his next six games .",
            "marker": "before",
            "output": ('Sanford started 12 consecutive games .', 'Steve Mason made his next start')
        },
        {
            "sentence": "Unlike before , the rumors were about player moves rather than coaching changes .",
            "previous_sentence": "With the losing continuing , more rumors began to surface .",
            "marker": "before",
            "output": None
        },
        {
            "sentence": "The room was further modified by two additions which gave it its current name : a sundial , and a delicate but sophisticated <unk> which was fixed to the ceiling of the Meridian Hall .",
            "previous_sentence": "The Sundial Room , also called the Meridian Hall , was once the residence of Queen Christina of Sweden , then newly converted to Catholicism .",
            "marker": "but",
            "output": None
        },
        {
            "sentence": "Some of these images , such as stars and cattle , are reminiscent of important features of Egyptian religion in later times , but in most cases there is not enough evidence to say whether the images are connected with deities .",
            "previous_sentence": "Predynastic artwork depicts a variety of animal and human figures .",
            "marker": "but",
            "output": ('Some of these images , such as stars and cattle , are reminiscent of important features of Egyptian religion in later times , .', 'in most cases there is not enough evidence to say whether the images are connected with deities')
        },
        {
            "sentence": "But in general , morality was based on practical ways to uphold maat in daily life , rather than on strict rules that the gods laid out .",
            "previous_sentence": "For example , the gods judged humans ' moral righteousness after death , and by the New Kingdom , a verdict of innocence in this judgment was believed to be necessary for admittance into the afterlife .",
            "marker": "but",
            "output": ("For example , the gods judged humans ' moral righteousness after death , and by the New Kingdom , a verdict of innocence in this judgment was believed to be necessary for admittance into the afterlife .", 'in general , morality was based on practical ways to uphold maat in daily life , rather than on strict rules that the gods laid out .')
        },
        {
            "sentence": "The CAA concludes that a severe reduction in GA would give \" some merit to the argument that pilot recruitment would be threatened \" , but that the data on flying hours \" does not support such a gloomy outlook . \"",
            "previous_sentence": "The counter argument to this claim is that pilots can be trained outside of the UK , and that the airline industry is not therefore dependent on a healthy GA sector in the UK for its supply of pilots .",
            "marker": "but",
            "output": ('a severe reduction in GA would give " some merit to the argument that pilot recruitment would be threatened " ,', 'the data on flying hours " does not support such a gloomy outlook')
        },
        {
            "sentence": "The use of chiral sulfides in a stoichiometric fashion has proved more successful than the corresponding catalytic variants , but the substrate scope is still limited in all cases .",
            "previous_sentence": "yielding an enantiomeric excess , which is labelled as \" ee \" ) variant of the Johnson – Corey – Chaykovsky reaction remains an active area of academic research .",
            "marker": "but",
            "output": ('The use of chiral sulfides in a stoichiometric fashion has proved more successful than the corresponding catalytic variants , .', 'the substrate scope is still limited in all cases')
        },
        {
            "sentence": "She was the victim of sexual abuse at the hands of the dictator himself , a sacrifice her father made to try to gain favor with the dictator again , a fact to which she alludes throughout the book , but which is only revealed at the very end : the book concludes with her recounting the memory of that night to her aunt and cousins , who never knew the true reason she left the country .",
            "previous_sentence": "The most apparent confrontation of memory is on the part of Urania Cabral , who has returned to the Dominican Republic for the first time in 30 years , and is forced to confront her father and the traumas that led her to leave the country at 14 .",
            "marker": "but",
            "output": None
        },
        {
            ## why is there no actual previous sentence here?
            "sentence": "But above all Mario Vargas Llosa uses the fictional Urania to facilitate the novel 's attempt at remembering the regime .",
            "previous_sentence": ".",
            "marker": "but",
            "output": ('.', "above all Mario Vargas Llosa uses the fictional Urania to facilitate the novel 's attempt at remembering the regime .")
        },
        {
            "sentence": "The populace may , for example , have mistaken the religion 's symbolic statements about the gods and their actions for literal truth .",
            "previous_sentence": "Commoners ' perceptions of the divine may have differed from those of the priests .",
            "marker": "for example",
            "output": ("Commoners ' perceptions of the divine may have differed from those of the priests .", "The populace may , , have mistaken the religion 's symbolic statements about the gods and their actions for literal truth .")
        },
        {
            "sentence": "Pre @-@ war civil aerodromes , for example Sywell , were returned to civilian use .",
            "previous_sentence": "This resulted in a significant inventory of facilities becoming available after the war .",
            "marker": "for example",
            "output": None
        },
        {
            "sentence": "Innis pointed out , for example , that as furs became scarce and trade in that staple declined , it became necessary to develop and export other staples such as wheat , potash and especially lumber .",
            "previous_sentence": "Innis theorized that this reliance on exporting natural resources made Canada dependent on more industrially advanced countries and resulted in periodic disruptions to economic life as the international demand for staples rose and fell ; as the staple itself became increasingly scarce ; and , as technological change resulted in shifts from one staple to others .",
            "marker": "for example",
            "output": ('Innis theorized that this reliance on exporting natural resources made Canada dependent on more industrially advanced countries and resulted in periodic disruptions to economic life as the international demand for staples rose and fell ; as the staple itself became increasingly scarce ; and , as technological change resulted in shifts from one staple to others .', 'Innis pointed out , , that as furs became scarce and trade in that staple declined , it became necessary to develop and export other staples such as wheat , potash and especially lumber .')
        },
        {
            "sentence": "Oxford Town \" , for example , was an account of James Meredith 's ordeal as the first black student to risk enrollment at the University of Mississippi .",
            "previous_sentence": "Many songs on this album were labeled protest songs , inspired partly by Guthrie and influenced by Pete Seeger 's passion for topical songs . \"",
            "marker": "for example",
            "output": ('Many songs on this album were labeled protest songs , inspired partly by Guthrie and influenced by Pete Seeger \'s passion for topical songs . "', 'Oxford Town " , , was an account of James Meredith \'s ordeal as the first black student to risk enrollment at the University of Mississippi .')
        },
        {
            "sentence": "Likewise , it is unclear whether prophylactic treatment of chronic infection is beneficial in persons who will undergo immunosuppression ( for example , organ transplant recipients ) or in persons who are already immunosuppressed ( for example , those with HIV infection ) .",
            "previous_sentence": "Treatment of chronic infection in women prior to or during pregnancy does not appear to reduce the probability the disease will be passed on to the infant .",
            "marker": "for example",
            "output": None
        },
        {
            ## I'm actually not sure whether this parse is correct... but the return value is a function of the parse for sure
            "sentence": "On 15 April for example , 800 Vietnamese men had been rounded up at the village of <unk> <unk> , tied together , executed , and their bodies dumped into the Mekong River .",
            "previous_sentence": "Cambodian soldiers and civilians then unleashed a reign of terror , murdering thousands of Vietnamese civilians .",
            "marker": "for example",
            "output": None
        },
        {
            "sentence": "However , not all of Alkan 's music is either lengthy or technically difficult ; for example , many of the Op.",
            "previous_sentence": ".",
            "marker": "for example",
            "output": None
        },
        {
            "sentence": "However , if the contestant used Jump the Question , they did not gain any money from the question they chose to skip ( for example , a contestant would not gain the typical $ 100 @,@ 000 if they jumped to the $ 250 @,@ 000 question ) .",
            "previous_sentence": "At any point prior to selecting a final answer , a contestant could use Jump the Question to skip the current question and move on to the next one , thus reducing the number of questions they had to correctly answer .",
            "marker": "for example",
            "output": None
        },
        {
            ## this is fine...
            "sentence": "<unk> are scarce in Scotland , Carolina <unk> ( <unk> <unk> ) for example , having been found only in Lauderdale .",
            "previous_sentence": ".",
            "marker": "for example",
            "output": None
        },
        {
            "sentence": "This is short @-@ lived , however , as following Maximilian 's defeat , Dahau and Calamity Raven move to activate an ancient <unk> super weapon within the Empire , kept secret by their benefactor .",
            "previous_sentence": "Partly due to these events , and partly due to the major losses in manpower Gallia suffers towards the end of the war with the Empire , the Nameless are offered a formal position as a squad in the Gallian Army rather than serve as an anonymous shadow force .",
            "marker": "however",
            "output": ('Partly due to these events , and partly due to the major losses in manpower Gallia suffers towards the end of the war with the Empire , the Nameless are offered a formal position as a squad in the Gallian Army rather than serve as an anonymous shadow force .', "This is short-lived , , as following Maximilian 's defeat , Dahau and Calamity Raven move to activate an ancient <unk> super weapon within the Empire , kept secret by their benefactor .")
        },
        {
            "sentence": "On June 10 , 1991 , however , the District Court declared the statute unconstitutional , stating that it violated both the Fourteenth and Seventeenth Amendments due to the failure to ensure \" popular participation \" through the use of primary elections .",
            "previous_sentence": "The case was first heard in the United States District Court for the Eastern District of Pennsylvania , where , following oral arguments , the judge dismissed both Trinsey 's motion to remove Wofford and the Commonwealth 's motion to dismiss .",
            "marker": "however",
            "output": ("The case was first heard in the United States District Court for the Eastern District of Pennsylvania , where , following oral arguments , the judge dismissed both Trinsey 's motion to remove Wofford and the Commonwealth 's motion to dismiss .", 'On June 10 , 1991 , , the District Court declared the statute unconstitutional , stating that it violated both the Fourteenth and Seventeenth Amendments due to the failure to ensure " popular participation " through the use of primary elections .')
        },
        {
            "sentence": "Compared to pre @-@ war classes , the absence of Polish Jewish students was notable , as they were confined by the Nazi Germans to ghettos ; there was , however , underground Jewish education in the ghettos , often organized with support from Polish organizations like TON .",
            "previous_sentence": "It is estimated that in some rural areas , the educational coverage was actually improved ( most likely as courses were being organized in some cases by teachers escaped or deported from the cities ) .",
            "marker": "however",
            "output": ('Compared to pre-war classes , the absence of Polish Jewish students was notable , as they were confined by the Nazi Germans to ghettos ; .', 'there was , , underground Jewish education in the ghettos , often organized with support from Polish organizations like TON')
        },
        {
            "sentence": "However , they found that the specimen was <unk> , and thus A. lucaris was a nomen dubium . \"",
            "previous_sentence": "Paul and Carpenter stated that the type specimen of this species , YPM 1931 , was from a younger age than Allosaurus , and might represent a different genus .",
            "marker": "however",
            "output": ('Paul and Carpenter stated that the type specimen of this species , YPM 1931 , was from a younger age than Allosaurus , and might represent a different genus .', ', they found that the specimen was <unk> , and thus A. lucaris was a nomen dubium . "')
        },
        {
            "sentence": "However , it took five years for its popularity to be established with professional users .",
            "previous_sentence": "The speaker was poorly received and its commercial life was short .",
            "marker": "however",
            "output": ('The speaker was poorly received and its commercial life was short .', ', it took five years for its popularity to be established with professional users .')
        },
        {
            ## this is nonsense, but that's only because the sentence split was wonky
            "sentence": "However , Stevens assembled his games from those that had been published in chess magazines , rather than complete collections of all the games played in particular events .",
            "previous_sentence": "@ 1 % .",
            "marker": "however",
            "output": ('@ 1 % .', ', Stevens assembled his games from those that had been published in chess magazines , rather than complete collections of all the games played in particular events .')
        },
        {
            "sentence": "However , at one point the writers had deliberated ending the series with Ross and Rachel in \" a gray area of where they aren ’ t together , but we hint there ’ s a sense that they might be down the road . \"",
            "previous_sentence": "When it finally came time to write the series finale , \" The only thing [ Crane and Kauffman ] absolutely knew from very early on was that we had to get Ross and Rachel together , \" deciding , \" We had <unk> the audience around for 10 years with their ' will they or won ’ t they , ' and we didn ’ t see any advantage in frustrating them \" any longer .",
            "marker": "however",
            "output": ('When it finally came time to write the series finale , " The only thing [ Crane and Kauffman ] absolutely knew from very early on was that we had to get Ross and Rachel together , " deciding , " We had <unk> the audience around for 10 years with their \' will they or won \xe2\x80\x99 t they , \' and we didn \xe2\x80\x99 t see any advantage in frustrating them " any longer .', ', at one point the writers had deliberated ending the series with Ross and Rachel in " a gray area of where they aren \xe2\x80\x99 t together , but we hint there \xe2\x80\x99 s a sense that they might be down the road . "')
        },
        {
            ## i am ok with this
            "marker": "if",
            "output": ('During gameplay , characters will call out .', 'something happens to them , such as their health points ( HP ) getting low or being knocked out by enemy attacks'),
            "previous_sentence": "Up to nine characters can be assigned to a single mission .",
            "sentence": "During gameplay , characters will call out if something happens to them , such as their health points ( HP ) getting low or being knocked out by enemy attacks ."
        },
        {
            ## wrong parse for later "if"s
            "marker": "if",
            "output": ('The level of purification can be monitored using various types of gel electrophoresis .', "the desired protein 's molecular weight and isoelectric point are known , by spectroscopy if the protein has distinguishable spectroscopic features , or by enzyme assays if the protein has enzymatic activity"),
            "previous_sentence": "Various types of chromatography are then used to isolate the protein or proteins of interest based on properties such as molecular weight , net charge and binding affinity .",
            "sentence": "The level of purification can be monitored using various types of gel electrophoresis if the desired protein 's molecular weight and isoelectric point are known , by spectroscopy if the protein has distinguishable spectroscopic features , or by enzyme assays if the protein has enzymatic activity ."
        },
        {
            "marker": "if",
            "output": ('the lyrics would work for two people', 'he could find a female artist as a duet partner'),
            "previous_sentence": "In an interview with <unk> , Aldean revealed that the song was not originally presented to him as a duet .",
            "sentence": "Nevertheless , he and producer Michael Knox thought the lyrics would work for two people if he could find a female artist as a duet partner ."
        },
        {
            "marker": "if",
            "output": (', , then she would', "Juan Per\xc3\xb3n would not accept Franco 's invitation for a state visit to Spain"),
            "previous_sentence": "Therefore , a visit to Franco , with António Salazar of Portugal the last remaining west European authoritarian leaders in power , would be diplomatically frowned upon internationally .",
            "sentence": "Fraser and Navarro write that Eva decided that , if Juan Perón would not accept Franco 's invitation for a state visit to Spain , then she would ."
        },
        {
            "marker": "if",
            "output": None,
            "previous_sentence": "Many early AI programs used the same basic algorithm .",
            "sentence": "To achieve some goal ( like winning a game or proving a theorem ) , they proceeded step by step towards it ( by making a move or a deduction ) as if searching through a maze , backtracking whenever they reached a dead end ."
        },
        {
            "marker": "if",
            "output": ('it is expected to sublime', 'it is exposed directly to solar radiation'),
            "previous_sentence": ".",
            "sentence": "Surface water ice is unstable at distances less than 5 AU from the Sun , so it is expected to sublime if it is exposed directly to solar radiation ."
        },
        {
            ## i'm OK with this, but really we should get rid of the "even"
            "marker": "if",
            "output": ('it important to promote new British artists ,', 'even it meant illustrations which some readers considered pornographic or offensive'),
            "previous_sentence": ".",
            "sentence": "When Samuel Carter Hall was choosing works to illustrate his newly launched The Art Journal , he considered it important to promote new British artists , even if it meant illustrations which some readers considered pornographic or offensive ."
        },
        {
            "marker": "if",
            "output": None,
            "previous_sentence": "In a book of the craters and boulders of Mars , Gibbons discovered a photograph of the Galle crater , which resembles a happy face , which they worked into an issue .",
            "sentence": "Moore said , \" We found a lot of these things started to generate themselves as if by magic \" , in particular citing an occasion where they decided to name a lock company the \" Gordian Knot Lock Company \" ."
        },
        {
            "marker": "meanwhile",
            "output": ('Egyptian cults sometimes incorporated Greek language , philosophy , iconography , and even temple architecture .', ', the cults of several Egyptian deities \xe2\x80\x94 particularly Isis , Osiris , Anubis , the form of Horus named Harpocrates , and the fused Greco-Egyptian god Serapis \xe2\x80\x94 were adopted into Roman religion and spread across the Roman Empire .'),
            "previous_sentence": "Egyptian cults sometimes incorporated Greek language , philosophy , iconography , and even temple architecture .",
            "sentence": "Meanwhile , the cults of several Egyptian deities — particularly Isis , Osiris , Anubis , the form of Horus named Harpocrates , and the fused Greco @-@ Egyptian god Serapis — were adopted into Roman religion and spread across the Roman Empire ."
        },
        {
            "marker": "meanwhile",
            "output": ('They write in an incomprehensible language , and the number of Observer sightings has increased over the past few months , leaving the team wondering why .', ', August visits the other Observers , who are not happy that he saved a girl who was supposed to die in the plane crash .'),
            "previous_sentence": "They write in an incomprehensible language , and the number of Observer sightings has increased over the past few months , leaving the team wondering why .",
            "sentence": "Meanwhile , August visits the other Observers , who are not happy that he saved a girl who was supposed to die in the plane crash ."
        },
        {
            "marker": "meanwhile",
            "output": ('In the episode , Cerie Xerox ( Bowden ) gets engaged so that she can be a " young hot mom " , causing Liz Lemon ( Tina Fey ) to think about marriage and having a family .', ", Jack Donaghy ( Alec Baldwin ) has trouble dealing with his own domineering mother , who wants to move in with him , and at the same time , Tracy Jordan ( Tracy Morgan ) becomes upset by Josh Girard 's ( Ross ) impression of him ."),
            "previous_sentence": "In the episode , Cerie Xerox ( Bowden ) gets engaged so that she can be a \" young hot mom \" , causing Liz Lemon ( Tina Fey ) to think about marriage and having a family .",
            "sentence": "Meanwhile , Jack Donaghy ( Alec Baldwin ) has trouble dealing with his own domineering mother , who wants to move in with him , and at the same time , Tracy Jordan ( Tracy Morgan ) becomes upset by Josh Girard 's ( Ross ) impression of him ."
        },
        {
            "marker": "meanwhile",
            "output": ('In the episode , Mulder is shown evidence of alien life which may actually be part of a huge government hoax designed to deflect attention from secret military programs .', ', Scully struggles with her cancer in the face of hostility from her brother , who believes she should no longer be working .'),
            "previous_sentence": "In the episode , Mulder is shown evidence of alien life which may actually be part of a huge government hoax designed to deflect attention from secret military programs .",
            "sentence": "Meanwhile , Scully struggles with her cancer in the face of hostility from her brother , who believes she should no longer be working ."
        },
        {
            "marker": "meanwhile",
            "output": ('Shekhar is disgusted to hear of the marriage and in his anger he scowls at Lolita , humiliating her like his father .', ", Girish assists Gurcharan 's family and takes them to London for the heart treatment ."),
            "previous_sentence": "Shekhar is disgusted to hear of the marriage and in his anger he scowls at Lolita , humiliating her like his father .",
            "sentence": "Meanwhile , Girish assists Gurcharan 's family and takes them to London for the heart treatment ."
        },
        {
            "marker": "meanwhile",
            "output": ('At the other end , Johnston was also able to make the ball move sideways .', ', Compton appeared to be untroubled by the bowling .'),
            "previous_sentence": "At the other end , Johnston was also able to make the ball move sideways .",
            "sentence": "Meanwhile , Compton appeared to be untroubled by the bowling ."
        },
        {
            "marker": "meanwhile",
            "output": ("In the episode , Peter and Lois go on a second honeymoon to spice up their marriage , but are chased by Mel Gibson after Peter steals the sequel to The Passion of the Christ from Gibson 's private hotel room .", ', Brian and Stewie take care of Chris and Meg at home .'),
            "previous_sentence": "In the episode , Peter and Lois go on a second honeymoon to spice up their marriage , but are chased by Mel Gibson after Peter steals the sequel to The Passion of the Christ from Gibson 's private hotel room .",
            "sentence": "Meanwhile , Brian and Stewie take care of Chris and Meg at home ."
        },
        {
            "marker": "meanwhile",
            "output": ('.', ", Chris learns that his best customer , Herbert , has made Kyle , a neighbor 's son and a bully , his new paper boy ."),
            "previous_sentence": ".",
            "sentence": "Meanwhile , Chris learns that his best customer , Herbert , has made Kyle , a neighbor 's son and a bully , his new paper boy ."
        },
        {
            "marker": "meanwhile",
            "output": ('When he informs George they cannot go back to his apartment because of the baby shower , George insists and Jerry discovers that George is wearing the red shirt and is just using the favor as a chance to encounter Leslie at the baby shower .', ', Kramer and the two Russians crash the shower to install cable television , start to eat all the food and get into a heated argument .'),
            "previous_sentence": "When he informs George they cannot go back to his apartment because of the baby shower , George insists and Jerry discovers that George is wearing the red shirt and is just using the favor as a chance to encounter Leslie at the baby shower .",
            "sentence": "Meanwhile , Kramer and the two Russians crash the shower to install cable television , start to eat all the food and get into a heated argument ."
        },
        {
            "marker": "meanwhile",
            "output": ('.', ', at 23 : 44 Tanaka ordered his ships to break contact and retire from the battle area .'),
            "previous_sentence": ".",
            "sentence": "Meanwhile , at 23 : 44 Tanaka ordered his ships to break contact and retire from the battle area ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "While at times this works to their advantage , such as a successful incursion into Imperial territory , other orders cause certain members of the 422nd great distress .",
            "sentence": "One such member , <unk> , becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven , attached to the ideal of Darcsen independence proposed by their leader , Dahau ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "The battle resulted in the capture of Kan Joy Chitam II of Palenque and made Toniná the dominant centre in the lower Usumacinta region .",
            "sentence": "The victory was so complete that it resulted in a ten @-@ year gap in the dynastic history of the defeated city , during which the captured ruler may have been held hostage ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "While he enjoyed the song 's concept , calling it \" so different and sinister that it 's more intriguing than the rest of the album \" , he felt that it was \" annoyingly displaced . \"",
            "sentence": "Hermann , on the other hand , called the track \" clever \" and \" spooky \" with \" music ... so well crafted that [ the concept ] works \" ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "Carey recorded a remix to the song as well , featuring hip @-@ hop verses from O.D.B of the Wu @-@ Tang Clan , as well as production from Puffy .",
            "sentence": "She spoke highly of the remix , complimenting Puffy and O.D.B , \" He 's so known in the street , and he 's one of the best people out there ... we kind of did what we both do and having O.D.B took it to another level ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "However , he claims to have been reading his paper at the time .",
            "sentence": "Also , it has been said that Young was too far away to identify the person in the boat , and that it couldn 't have been Carol Park 's body that was being dumped , as the Youngs were positioned so that the location Carol 's body was found would have been visibly blocked by an island ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "Humpty Dumpty is a character in an English nursery rhyme , probably originally a riddle and one of the best known in the English @-@ speaking world .",
            "sentence": "He is typically portrayed as an anthropomorphic egg , though he is not explicitly described so ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "The choreography was done by Danielle Polanco and <unk> ' Moaning , who used a 1980 ’ s retro set .",
            "sentence": "Beyoncé explained the concept of the video at MTV : \" It 's probably the most flamboyant video , and the metallic dresses are so beautiful , they added so much color ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "Khánh then asked his colleagues to participate in a campaign of fomenting anti @-@ American street protests and to give the impression the country did not need Washington 's aid .",
            "sentence": "A CIA informant reported the recent arguments with Taylor had incensed the volatile Thi so much that he had privately vowed to \" blow up everything \" and \" kill Phan Khắc Sửu , Trần Văn Hương and Nguyễn Khánh and put an end to all this ."
        },
        {
            "marker": "so",
            "output": None,
            "previous_sentence": "Three years later , Spain revoked the treaty and in 1800 secretly returned Louisiana to Napoleonic France in the Third Treaty of San Ildefonso .",
            "sentence": "This transfer was so secret that the Spanish continued to administer the territory ."
        },
        {
            "sentence": "The government buried many in mass graves , some above @-@ ground tombs were forced open so bodies could be stacked inside , and others were burned .",
            "previous_sentence": "Mayor Jean @-@ Yves Jason said that officials argued for hours about what to do with the volume of corpses .",
            "output": ('some above-ground tombs were forced open', 'bodies could be stacked inside'),
            "marker": "so"
        },
        {
            ## i'm ok with this
            "sentence": "This study includes Auroraceratops , but lacks seven taxa found in Xu and Makovicky 's work , so it is unclear how comparable the two studies are .",
            "previous_sentence": "In contrast to the previous analysis , You and Dodson find Chaoyangsaurus to be the most basal neoceratopsian , more derived than Psittacosaurus , while Leptoceratopsidae , not Protoceratopsidae , is recovered as the sister group of Ceratopsidae .",
            "output": ("lacks seven taxa found in Xu and Makovicky 's work ,", 'it is unclear how comparable the two studies are'),
            "marker": "so"
        },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": "Its first public appearance was at the 2010 Tokyo Game Show ( TGS ) , where a demo was made available for journalists and attendees .",
        #     "sentence": "During the publicity , story details were kept scant so as not to spoil too much for potential players , along with some of its content still being in flux at the time of its reveal ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": "Jack finally agrees to pay for Ernest , everyone thinking that it is Algernon 's bill when in fact it is his own .",
        #     "sentence": "The four @-@ act version was first played on the radio in a BBC production and is still sometimes performed ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": "A protective wall of massive proportions surrounded the cathedral precinct , but only a small section has survived .",
        #     "sentence": "The wall had four access gates , one of which — the Pans Port — still exists ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": ".",
        #     "sentence": "We are still pursuing but it has been perforce slow as the horses are done and the enemy , when advancing , entrenched himself at various points … which has enabled him to fight a most masterly rearguard action … As I am moving on , I must close ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": "Khánh did not want his rival taking power , so he and the Americans convinced the HNC to dilute the powers of the position to make it unappealing to Minh , who was then sent on an overseas diplomatic goodwill tour to remove him from the political scene .",
        #     "sentence": "However , Minh was back in South Vietnam after a few months and the power balance in the junta was still fragile ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": "Jared Johnson of Allmusic gave the album four out of five stars and described it as a \" powerful worship experience \" , but also stated that \" some might wonder how a little more variety would sound from such experienced professionals ... the band 's core sound continues to land in the AC cross hairs \" .",
        #     "sentence": "Andrew Greer of CCM Magazine commented that the album \" ups the musical ante a bit , with some borderline poetic verses and interesting musical riffs \" , but also commented that \" many of these tracks still suffer from the ' Crowns Cliché Syndrome , ' using lyrical Christian @-@ ese to produce trite rhymes that seem hard @-@ pressed to energize a ready @-@ to @-@ worship crowd \" ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": "With a mass only 80 times that of Jupiter ( MJ ) , 2MASS <unk> @-@ 1403 is the smallest known star undergoing nuclear fusion in its core .",
        #     "sentence": "For stars with metallicity similar to the Sun , the theoretical minimum mass the star can have and still undergo fusion at the core , is estimated to be about 75 MJ ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": "He was selected by the Arizona Diamondbacks in the fourth round of the 1998 Major League Baseball ( MLB ) Draft .",
        #     "sentence": "He began his career as a starting pitcher but struggled and converted into a sidearm ( or submarine ) reliever while still in the Diamondbacks ' organization ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": ".",
        #     "sentence": "As of 2014 , Spokane is still trying to make the transition to a more service @-@ oriented economy in the face of a less prominent manufacturing sector ."
        # },
        # {
        #     "marker": "still",
        #     "output": None,
        #     "previous_sentence": "Quentin Tarantino as Himself ( Extended version ) : In a short appearance with Kermit the Frog , Tarantino discusses ideas on how to stop the Wicked Witch of the West .",
        #     "sentence": "Despite the fact that his role is small his name is still mentioned in the movie trailer and listed on the cover of both the Video and DVD ."
        # },
        {
            "marker": "then",
            "output":  ('Now owing allegiance to none other than themselves , the 422nd confronts Dahau and destroys the <unk> weapon .', 'Each member goes their separate ways in order to begin their lives anew .'),
            "previous_sentence": "Now owing allegiance to none other than themselves , the 422nd confronts Dahau and destroys the <unk> weapon .",
            "sentence": "Each member then goes their separate ways in order to begin their lives anew ."
        },
        {
            "marker": "then",
            "output": ("Bannawit denied that his own transfer was the result of his criticism of Saprang 's transfer .", 'Bannawit announced that he would resign from the military and enter politics .'),
            "previous_sentence": "Bannawit denied that his own transfer was the result of his criticism of Saprang 's transfer .",
            "sentence": "Bannawit then announced that he would resign from the military and enter politics ."
        },
        {
            "marker": "then",
            "output": None,
            "previous_sentence": "On July 6 , 1916 , Romania declared war on Germany and Austria @-@ Hungary , following the initial success of the Brusilov Offensive ( a major Russian offensive against the armies of the Central Powers on the Eastern Front ) .",
            "sentence": "The Romanian armies entered Transylvania ( then part of the Austro @-@ Hungarian Empire ) , together with Russian forces ."
        },
        {
            "marker": "then",
            "output": None,
            "previous_sentence": ".",
            "sentence": "Montgomery 's plan was for the Canadian Division to attack across the Moro in the coastal lowlands to take Ortona first and then Pescara ."
        },
        {
            "marker": "then",
            "output": None,
            "previous_sentence": "Van der Weyden 's depiction of the Magdalen is based on Mary of Bethany , identified by the time of Pope Gregory I as the repentant prostitute of Luke 7 : 36 – 50 .",
            "sentence": "She then became associated with weeping and reading : Christ 's mercy causes the eyes of the sinner to be contrite or tearful ."
        },
        {
            "marker": "then",
            "output": ('Carol was killed due to blunt trauma to her face by means of some instrument , alleged in court to have been an ice axe .', 'She was bound with rope , using complex knots , weighed down with rocks and lead pipes and thrown overboard from a boat on Coniston Water .'),
            "previous_sentence": "Carol was killed due to blunt trauma to her face by means of some instrument , alleged in court to have been an ice axe .",
            "sentence": "She was then bound with rope , using complex knots , weighed down with rocks and lead pipes and thrown overboard from a boat on Coniston Water ."
        },
        {
            "marker": "then",
            "output": ("The previous removal of Evita 's body was avenged by the Montoneros when they in 1970 stole Pedro Eugenio Aramburu 's corpse , whom they had previously killed .", "Montoneros used the captive body of Aramburu to pressure for the repatriation of Evita 's body ."),
            "previous_sentence": "The previous removal of Evita 's body was avenged by the Montoneros when they in 1970 stole Pedro Eugenio Aramburu 's corpse , whom they had previously killed .",
            "sentence": "Montoneros then used the captive body of Aramburu to pressure for the repatriation of Evita 's body ."
        },
        {
            "marker": "then",
            "output": ('Bart decides that Burns is his true father and the two celebrate by firing employees at the Springfield Nuclear Power Plant by ejecting them through a trap door .', "However , one of the employees is Homer and Mr. Burns tries to break Bart 's ties with his family by forcing him to fire Homer ."),
            "previous_sentence": "Bart decides that Burns is his true father and the two celebrate by firing employees at the Springfield Nuclear Power Plant by ejecting them through a trap door .",
            "sentence": "However , one of the employees is Homer and Mr. Burns then tries to break Bart 's ties with his family by forcing him to fire Homer ."
        },
        {
            "marker": "then",
            "output": None,
            "previous_sentence": ".",
            "sentence": "The active constituents of this species are water @-@ soluble , and boiling and then discarding the cooking water at least partly <unk> A. muscaria ."
        },
        {
            "marker": "though",
            "output": (", it wasn ' t until 1985 that a winter collection was assembled from her remaining work and published posthumously .", 'she published Flower Fairy books with spring , summer , and autumn themes'),
            "previous_sentence": "Barker died in 1973 .",
            "sentence": "Though she published Flower Fairy books with spring , summer , and autumn themes , it wasn 't until 1985 that a winter collection was assembled from her remaining work and published posthumously ."
        },
        {
            "marker": "though",
            "output": None,
            "previous_sentence": "By the close of the 19th century , the fort was largely unused and had no defence significance .",
            "sentence": "It was briefly used for military purposes during World War I and World War II , though not for its original defensive role ."
        },
        {
            "marker": "though",
            "output": ('Occupancy at 100 McAllister was low , .', 'the United States Army Corps of Engineers moved their San Francisco District offices there in the 1960s , and local draftees were still required to appear there through the late 1960s'),
            "previous_sentence": "Many federal groups at 100 McAllister moved their offices in 1959 – 1960 to the newly built federal building at 450 Golden Gate Avenue , later named the Phillip Burton Federal Building .",
            "sentence": "Occupancy at 100 McAllister was low , though the United States Army Corps of Engineers moved their San Francisco District offices there in the 1960s , and local draftees were still required to appear there through the late 1960s ."
        },
        {
            ## i'm OK with this, but really we should get rid of the "even"
            "marker": "though",
            "output": ('This plan was discarded since local residents feared noise pollution and destroyed lawns .', 'Even Berg was not made a station on the Ring Line , noise shields were put up along the Sognsvann Line .'),
            "previous_sentence": "This plan was discarded since local residents feared noise pollution and destroyed lawns .",
            "sentence": "Even though Berg was not made a station on the Ring Line , noise shields were put up along the Sognsvann Line ."
        },
        {
            "marker": "though",
            "output": ('Many etymological suggestions have been made , : .', 'there is no general agreement'),
            "previous_sentence": ".",
            "sentence": "Many etymological suggestions have been made , though there is no general agreement : ."
        },
        {
            "marker": "though",
            "output": ('<unk> ) , described by Hulke two years after I. prestwichii , has been synonymised with Iguanodon bernissartensis , .', 'this is controversial'),
            "previous_sentence": "<unk> ( also incorrectly spelled I.",
            "sentence": "<unk> ) , described by Hulke two years after I. prestwichii , has been synonymised with Iguanodon bernissartensis , though this is controversial ."
        },
        {
            "marker": "though",
            "output": (', no effective antibacterial treatments were available .', 'it was known in the nineteenth century that bacteria are the cause of many diseases'),
            "previous_sentence": ".",
            "sentence": "Though it was known in the nineteenth century that bacteria are the cause of many diseases , no effective antibacterial treatments were available ."
        },
        {
            "marker": "though",
            "output": (', he was credited as a supervising producer on several installments for the third season .', 'Craig left the staff shortly before " Oubliette " entered production for unknown reasons'),
            "previous_sentence": "His most notable television screenplay was The New Alfred Hitchcock Presents entry \" Final Escape \" .",
            "sentence": "Though Craig left the staff shortly before \" Oubliette \" entered production for unknown reasons , he was credited as a supervising producer on several installments for the third season ."
        },
        {
            "marker": "though",
            "output": (', Pennock did not appear in either series .', 'the Red Sox won the 1915 and 1916 World Series'),
            "previous_sentence": "@ 67 ERA , as Buffalo won the league pennant .",
            "sentence": "Though the Red Sox won the 1915 and 1916 World Series , Pennock did not appear in either series ."
        },
        {
            "marker": "when",
            "output": (', the other is sealed off to the player', 'one option is selected'),
            "previous_sentence": "The player progresses through a series of linear missions , gradually unlocked as maps that can be freely scanned through and replayed as they are unlocked .",
            "sentence": "The route to each story location on the map varies depending on an individual player 's approach : when one option is selected , the other is sealed off to the player ."
        },
        {
            "marker": "when",
            "output": ('The team received help , however , .', 'Jordan decided to return to the NBA for the Bulls'),
            "previous_sentence": "Struggling at mid @-@ season to ensure a spot in the playoffs , Chicago was 31 – 31 at one point in mid @-@ March .",
            "sentence": "The team received help , however , when Jordan decided to return to the NBA for the Bulls ."
        },
        {
            "marker": "when",
            "output": ('The storyline began on-screen .', 'Nicole started dating Elliot Gillen ( Paul <unk> )'),
            "previous_sentence": ".",
            "sentence": "The storyline began on @-@ screen when Nicole started dating Elliot Gillen ( Paul <unk> ) ."
        },
        {
            "marker": "when",
            "output": (', he speaks in " Square mouth public talk " , which is a manner of speaking reserved for martial heroes , highly respected characters , or , sometimes , lesser characters that pretend to be an important hero .', 'Zhou is vocalized in " Yangzhou storytelling "'),
            "previous_sentence": ".",
            "sentence": "When Zhou is vocalized in \" Yangzhou storytelling \" , he speaks in \" Square mouth public talk \" , which is a manner of speaking reserved for martial heroes , highly respected characters , or , sometimes , lesser characters that pretend to be an important hero ."
        },
        {
            "marker": "when",
            "output": None,
            "previous_sentence": "Ramblin ' Man \" features a conversation between an interviewer ( the voice of Michael Deakin — father of Lemon Jelly 's Fred Deakin ) and \" John the Ramblin ' Man \" ( the voice of Standing ) , during which he lists various places from around the world , ranging from \" from small Sussex villages to major world capitals . \"",
            "sentence": "When listed in the order in which the locations are narrated , the message \" Bagpuss Sees All Things \" is spelled out midway through the song ( from Brixton at four minutes ten seconds , to San José at four minutes 31 seconds ) using the first letter of each location ."
        },
        {
            "marker": "when",
            "output": (', the family moved to Coleraine , County Londonderry , where May worked for the Housing Executive .', 'Nesbitt was 11 years old'),
            "previous_sentence": ".",
            "sentence": "When Nesbitt was 11 years old , the family moved to Coleraine , County Londonderry , where May worked for the Housing Executive ."
        },
        {
            "marker": "when",
            "output": None,
            "previous_sentence": ".",
            "sentence": "The following managers have all won at least one trophy when in charge or have been notable for Villa in the context of the League , for example Jozef Vengloš who holds a League record ."
        },
        {
            "marker": "when",
            "output": (', he had Tintin befriend a herd of elephants living in the Indian jungle', 'he wrote Cigars of the Pharaoh ( 1934 )'),
            "previous_sentence": "Such scenes reflect the popularity of big @-@ game hunting among whites and affluent visitors in Sub @-@ Saharan Africa during the 1930s .",
            "sentence": "Hergé later felt guilty about his portrayal of animals in Tintin in the Congo and became an opponent of blood sports ; when he wrote Cigars of the Pharaoh ( 1934 ) , he had Tintin befriend a herd of elephants living in the Indian jungle ."
        },
        {
            "marker": "when",
            "output": ('the rock fainted', 'it was produced in court'),
            "previous_sentence": "Professor Kenneth Pye , a defense witness , said that there was no evidence that the rock had ever been on the lake bed at all .",
            "sentence": "The article in the Sunday Herald also claims that the policeman said to be responsible for finding the rock fainted when it was produced in court , offering no explanation , but still denying that he ever found it ."
        },
        {
            "marker": "while",
            "output": (', it also underwent multiple adjustments , such as making the game more forgiving for series newcomers .', 'it retained the standard features of the series'),
            "previous_sentence": "The game began development in 2010 , carrying over a large portion of the work done on Valkyria Chronicles II .",
            "sentence": "While it retained the standard features of the series , it also underwent multiple adjustments , such as making the game more forgiving for series newcomers ."
        },
        {
            "marker": "while",
            "output": (', all populations show great individual variation in colouring , and the birds gradually become paler and greyer towards the east of the range .', 'there are no subspecies'),
            "previous_sentence": "The chicks have black down , as with all rails .",
            "sentence": "While there are no subspecies , all populations show great individual variation in colouring , and the birds gradually become paler and greyer towards the east of the range ."
        },
        {
            "marker": "while",
            "output": ('to put punk ideology in the series , with the protagonist trying to reacquire his former punk self ,', 'also characterizing the Conservative government as a demon infestation with the punk subculture fighting against this supposed subversion and abuse'),
            "previous_sentence": "During Warren Ellis ' run , he included American school shootings in a one @-@ shot issue which led to a major controversy .",
            "sentence": "In his run , Peter Milligan managed to put punk ideology in the series , with the protagonist trying to reacquire his former punk self , while also characterizing the Conservative government as a demon infestation with the punk subculture fighting against this supposed subversion and abuse ."
        },
        {
            "marker": "while",
            "output": None,
            "previous_sentence": "Yankovic , however , directed one himself which was mostly made up of stock footage , with a live action finale that was filmed in an economically devastated part of the Bronx , New York that looked like a bomb had gone off .",
            "sentence": "The final original that was recorded was \" Good Enough for Now \" , a country music pastiche about how the singer 's lover , who , while not the best , will do for now ."
        },
        {
            "marker": "while",
            "output": ('The Sedgemoor district has many buildings related to trade and commerce centered on Bridgwater ; .', 'in South Somerset abbeys , priories and farmhouses predominate'),
            "previous_sentence": "North Somerset features bridges and piers along with a selection of Manor houses .",
            "sentence": "The Sedgemoor district has many buildings related to trade and commerce centered on Bridgwater ; while in South Somerset abbeys , priories and farmhouses predominate ."
        },
        {
            ## I am OK with this
            "marker": "while",
            "output": ('to wait for the rear', 'the main army marched on to Castleton'),
            "previous_sentence": "American general St. Clair paused at Hubbardton to give the main army 's tired and hungry troops time to rest while he hoped the rear guard would arrive .",
            "sentence": "When it did not arrive in time , he left Colonel Seth Warner and the Green Mountain Boys behind , along with the 2nd New Hampshire Regiment under Colonel Nathan Hale , at Hubbardton to wait for the rear while the main army marched on to Castleton ."
        },
        {
            "marker": "while",
            "output": None,
            "previous_sentence": "Towards the end of the song , Carey belts out the climax .",
            "sentence": "Sarah Rodman from The Boston Herald described it as fascinating and wrote , \" it sounds as though Carey is singing in falsetto while still in her chest voice . \""
        },
    ]
    curious_cases = [
        {
            ## wrong parse
            "marker": "when",
            "output": None,
            "previous_sentence": "At the time , NCAA guidelines for bowls required 75 percent of gross receipts to go to participating schools , with 33 percent of tickets to the game also required to go to each school .",
            "sentence": "In 1983 , the NCAA threatened to revoke the Peach Bowl 's charter when ticket sales hovered around 25 @,@ 000 with a week to go before the bowl ."
        },
        {
            "sentence": "But , after inspecting the work and observing the spirit of the men I decided that a garrison 500 strong could hold out against Fitch and that I would lead the remainder - about 1500 - to Gen 'l Rust as soon as shotguns and rifles could be obtained from Little Rock instead of pikes and lances , with which most of them were armed .",
            "previous_sentence": "",
            "marker": "after",
            "output": None,
            "explanation": "incorrect parse. it thinks 'the men I decided ...' forms a relative clause. different from what's up at http://nlp.stanford.edu:8080/corenlp/process"
        },
        {
            "sentence": "In 1864 , after Little Rock fell to the Union Army and the arsenal had been recaptured , General Fredrick Steele marched 8 @,@ 500 troops from the arsenal beginning the Camden Expedition .",
            "previous_sentence": "NA",
            "marker": "after",
            "output": None
        },
        {
            "sentence": "In addition to Sega staff from the previous games , development work was also handled by <unk> The original scenario was written Kazuki Yamanobe , while the script was written by Hiroyuki Fujii , Koichi Majima , <unk> Miyagi , Seiki <unk> and Takayuki <unk> .",
            "previous_sentence": "Speaking in an interview , it was stated that the development team considered Valkyria Chronicles III to be the series ' first true sequel : while Valkyria Chronicles II had required a large amount of trial and error during development due to the platform move , the third game gave them a chance to improve upon the best parts of Valkyria Chronicles II due to being on the same platform .",
            "marker": "also",
            "output": ('while the script was written by Hiroyuki Fujii , Koichi Majima , <unk> Miyagi , Seiki <unk> and Takayuki <unk>', 'In addition to Sega staff from the previous games , development work was handled by <unk> The original scenario was written Kazuki Yamanobe , .')
        },
        {
            ## wrong parse :(
            "marker": "so",
            "output": None,
            "previous_sentence": ".",
            "sentence": "The developers designed the game like a \" Japanese garden \" , where they attempted to remove all of the game elements that did not fit with the others , so the emotions they wanted the game to evoke would come through ."
        },
        {
            ## weird parse - in particular "so" has an uncategorized dependency to S1 rather than a mark dependency to S2...
            "sentence": "Nicole felt she was too young and unable to offer a child stability , so she agreed to let Marilyn Chambers adopt the baby upon its birth .",
            "previous_sentence": "One of her final storylines was a pregnancy plot .",
            "marker": "so",
            "output": None
        },
        ## parse is just wrong :(
        {
            "sentence": "There are also love simulation elements related to the game 's two main heroines , although they take a very minor role .",
            "previous_sentence": "After the game 's completion , additional episodes are unlocked , some of them having a higher difficulty than those found in the rest of the game .",
            "marker": "although",
            "output": ("love simulation elements related to the game 's two main heroines ,", 'they take a very minor role')
        },
        {
            "sentence": "The remainder held professional pilot licences , either a Commercial Pilot Licence or an Airline Transport Pilot Licence , although not all of these would be engaged in GA activities .",
            "previous_sentence": "The number of pilots licensed by the CAA to fly powered aircraft in 2005 was 47 @,@ 000 , of whom 28 @,@ 000 held a Private Pilot Licence .",
            "marker": "although",
            "output": ('either a Commercial Pilot Licence or an Airline Transport Pilot Licence ,', 'not all of these would be engaged in GA activities')
        },
        {
        ## parse is just wrong :(
        ## it thinks "Calamity Raven move" is a compound NP
            "sentence": "This is short @-@ lived , however , as following Maximilian 's defeat , Dahau and Calamity Raven move to activate an ancient <unk> super weapon within the Empire , kept secret by their benefactor .",
            "previous_sentence": "Partly due to these events , and partly due to the major losses in manpower Gallia suffers towards the end of the war with the Empire , the Nameless are offered a formal position as a squad in the Gallian Army rather than serve as an anonymous shadow force .",
            "marker": "as",
            "output": None
        },
        ## parse is just wrong :(
        ## thought, to be fair, this sentence garden-pathed me into thinking the same thing as the parser did
        {
            "sentence": "As an armed Gallian force invading the Empire just following the two nations ' cease @-@ fire would certainly wreck their newfound peace , Kurt decides to once again make his squad the Nameless , asking Crowe to list himself and all under his command as killed @-@ in @-@ action .",
            "previous_sentence": "Without the support of Maximilian or the chance to prove themselves in the war with Gallia , it is Dahau 's last trump card in creating a new Darcsen nation .",
            "marker": "as",
            "output": None
        },
        ## perhaps attaches to the VP, not to "because",
        ## which could be correct for other sentences, but isn't right for this one
        {
            "sentence": "Perhaps because Abraham Lincoln had not yet been inaugurated as President , Captain Totten received no instructions from his superiors and was forced to withdraw his troops .",
            "previous_sentence": ".",
            "marker": "because",
            "output": (', Captain Totten received no instructions from his superiors and was forced to withdraw his troops .', 'Perhaps Abraham Lincoln had not yet been inaugurated as President')
        },
        {
            ## incorrect parse
            "marker": "while",
            "output": ('they considered Jordan their role model', 'growing up , including LeBron James and Dwyane Wade'),
            "previous_sentence": "Jordan 's athletic leaping ability , highlighted in his back @-@ to @-@ back slam dunk contest championships in 1987 and 1988 , is credited by many with having influenced a generation of young players .",
            "sentence": "Several current NBA All @-@ Stars have stated that they considered Jordan their role model while growing up , including LeBron James and Dwyane Wade ."
        },
        {
          ## wrong parse
            "marker": "then",
            "output": None,
            "previous_sentence": "No profit was made in six years , and the church left , losing their investment .",
            "sentence": "In the late 1930s the building housed the Empire Hotel , known for its Sky Room lounge , then from World War II to the 1970s , 100 McAllister served as U.S. government offices ."
        },
        {
            "sentence": "One of the primary reasons why Jordan was not drafted sooner was because the first two teams were in need of a center .",
            "previous_sentence": "The Chicago Bulls selected Jordan with the third overall pick , after Hakeem Olajuwon ( Houston Rockets ) and Sam Bowie ( Portland Trail Blazers ) .",
            "marker": "because",
            "output": ('One of the primary reasons why Jordan was not drafted sooner was .', 'the first two teams were in need of a center')
        },
        {
            ## not sure what could be done about this
            "sentence": "Hellblazer was first published during the early days of the Modern Age of Comics , and so its themes were dark , edgy , politically and morally complex as its contemporaries .",
            "previous_sentence": ".",
            "output": None,
            "marker": "so"
        },
        ## this honestly might be right. what does this sentence even mean?
        {
            "sentence": "The main theme was initially recorded using orchestra , then Sakimoto removed elements such as the guitar and bass , then adjusted the theme using a synthesizer before redoing segments such as the guitar piece on their own before incorporating them into the theme .",
            "previous_sentence": "He <unk> the main theme about seven times through the music production due to this need to reassess the game .",
            "marker": "before",
            "output": ('redoing segments such as the guitar piece on their own', 'incorporating them into the theme')
        },
        {
            "sentence": "Nothing of this sort had been before attempted on Government account in Arkansas to my knowledge , except for the manufacture of small arms , the machinery for which was taken away by General Van Dorn and there was neither capital nor sufficient enterprise among the citizens to engage in such undertakings <unk> A further supply , along with lead and caps , was procured from the citizens of Little Rock and vicinity by donation , purchases , and impressments .",
            "previous_sentence": "The tools , machinery , and the material were gathered piecemeal or else made by hand labor .",
            "marker": "before",
            "output": ('The tools , machinery , and the material were gathered piecemeal or else made by hand labor .', 'Nothing of this sort had been attempted on Government account in Arkansas to my knowledge , except for the manufacture of small arms , the machinery for which was taken away by General Van Dorn and there was neither capital nor sufficient enterprise among the citizens to engage in such undertakings <unk> A further supply , along with lead and caps , was procured from the citizens of Little Rock and vicinity by donation , purchases , and impressments .')
        },
        {
            ## wrong parse
            "marker": "while",
            "output": None,
            "previous_sentence": "Dahl was given a free rein on his script , except for the character of Bond and \" the girl formula \" , involving three women for Bond to seduce : an ally and a henchwoman who both get killed , and the main Bond girl .",
            "sentence": "While the third involved a character from the book , Kissy Suzuki , Dahl had to create Aki and Helga Brandt to fulfil the rest ."
        },
        ## parse is very wrong, and not only because the sentence is split badly and has a quote in it
        {
            "sentence": "Two days elapsed before the change could be effected . \"",
            "previous_sentence": "But , after inspecting the work and observing the spirit of the men I decided that a garrison 500 strong could hold out against Fitch and that I would lead the remainder - about 1500 - to Gen 'l Rust as soon as shotguns and rifles could be obtained from Little Rock instead of pikes and lances , with which most of them were armed .",
            "marker": "before",
            "output": ('elapsed', 'the change could be effected')
        },
        ## parse is weird...
        {
            "sentence": "Before any of these rumors came to fruition , the St. Louis Blues asked Columbus for permission to hire Hitchcock , which the Blue Jackets allowed .",
            "previous_sentence": "Hitchcock had previously coached the Blue Jackets to their only playoff appearance in club history and was still under contract with the franchise through to the end of the season .",
            "marker": "before",
            "output": None
        },
        {
        ## win is incorrectly parsed as a verb
            "sentence": "They extended the streak to four with a win over the Los Angeles Kings before it came to an end with a 4 – 1 loss to the St. Louis Blues .",
            "previous_sentence": "Columbus again defeated the Coyotes three days later to earn their first three @-@ game win streak of the season .",
            "marker": "before",
            "output": ('with a win over the Los Angeles Kings', 'it came to an end with a 4 \xe2\x80\x93 1 loss to the St. Louis Blues')
        },
        {
            ## parse is wrong
            "sentence": "Jones had by now moved on to another posting but Eaton took up the squadron 's command as planned .",
            "previous_sentence": "12 ( General Purpose ) Squadron was not formed until 6 February 1939 at Laverton .",
            "marker": "but",
            "output": ('by now moved on to another posting', "Eaton took up the squadron 's command as planned")
        },
        {
            ## wrong parse
            "sentence": "EW quoted Mindy Kaling as saying , \" I always feel unoriginal bringing up Tina as my inspiration , but she 's everyone 's inspiration for a reason . \"",
            "previous_sentence": "In 2013 , Entertainment Weekly crowned Fey as \" The Once and Future Queen \" ( an allusion to The Once and Future King ) in their feature on \" Women Who Run TV , \" calling her \" the funniest woman in the free world . \"",
            "marker": "but",
            "output": ('EW quoted Mindy Kaling as saying , " I always feel unoriginal bringing up Tina as my inspiration , . "', "she 's everyone 's inspiration for a reason")
        },
        {
            ## wrong parse, localized is an nmod on sales?
            "sentence": "Due to low sales of Valkyria Chronicles II , Valkyria Chronicles III was not localized , but a fan translation compatible with the game 's expanded edition was released in 2014 .",
            "previous_sentence": "It was also adapted into manga and an original video animation series .",
            "marker": "but",
            "output": None
        },
        {
            ## incorrect parse
            "sentence": "It 's taken me long enough to get to Hull so I 'm not going to walk out after a few months , or whatever . \"",
            "previous_sentence": "...",
            "output": None,
            "marker": "so"
        },
        {
            ## weird parse
            "sentence": "Of her departure , James said \" I was at Home and Away for three @-@ and @-@ a @-@ half @-@ years , so it 's good to be finished and get to be who I am , and do what I 've wanted to do for so long . \"",
            "previous_sentence": "She has already filmed her final scenes and Nicole will leave on @-@ screen later in the year .",
            "output": None,
            "marker": "so"
        },
        {
            ## slight parsing error
            "sentence": "Where a subadult fails to leave his maternal range , for example , he may be killed by his father .",
            "previous_sentence": "Because males disperse farther than females and compete more directly for mates and territory , they are most likely to be involved in conflict .",
            "marker": "for example",
            "output": None
        },
        {
            # totally incorrect parse because it thinks 1.e4 is a sentence barrier
            "sentence": "I do however believe that with either 1.e4 or 1.d4 , White should be able to obtain some sort of advantage that persists into the endgame .",
            "previous_sentence": "Kaufman writes , \" I don 't believe that White has a forced win in Chess .",
            "marker": "however",
            "output": None
        },
        {
            # incorrect parse: it thinks however is an adverb on misleading opponents
            "sentence": "In order to mislead his opponents however , he initially attacked and captured the towns of <unk> de Matamoros ( which was subsequently retaken by federal forces ) and <unk> .",
            "previous_sentence": "Subsequently Zapata , for political and strategic reasons , decided to attack the city of Cuautla .",
            "marker": "however",
            "output": None
        },
        {
            ## though is advmod here...
            "marker": "though",
            "output": None,
            "previous_sentence": "Horrified to learn that he was adopted after being discovered as a baby in a handbag at Victoria Station , she refuses him and forbids further contact with her daughter .",
            "sentence": "Gwendolen , though , manages covertly to promise to him her undying love ."
        },
        {
            # parses as ccomp, which i'm rejecting in favor of high precision
            "sentence": "Through her relationship with Geoff she had mellowed , however her vanity was still present .",
            "previous_sentence": "Nicole relates to the wild side of Freya , but has no idea how far Freya is going to take it . \"",
            "marker": "however",
            "output": ('Through her relationship with Geoff she had mellowed , .', 'her vanity was still present')
        },
        {
            ## i'm not sure what a good parse of this would be...
            "marker": "while",
            "output": ('White has a spatial advantage , . "', 'Black often maneuvers his pieces on the last two ranks of the board , but White " has to keep a constant eye on the possible liberating pawn thrusts ... b5 and ... d5'),
            "previous_sentence": "@ 0 @-@ 0 e6 6.Nc3 Be7 7.d4 cxd4 <unk> d6 <unk> a6 .",
            "sentence": "White has a spatial advantage , while Black often maneuvers his pieces on the last two ranks of the board , but White \" has to keep a constant eye on the possible liberating pawn thrusts ... b5 and ... d5 . \""
        },
        {
            ## wrong parse
            "marker": "if",
            "output": None,
            "previous_sentence": "The Catechism states that , with the help of God 's grace , men and women are required to overcome lust and bodily desires \" for sinful relationships with another person 's spouse . \"",
            "sentence": "In Theology of the Body , a series of lectures given by Pope John Paul II , Jesus ' statement in Matthew 5 : 28 is interpreted that one can commit adultery in the heart not only with another 's spouse , but also with his / her own spouse if one looks at him / her lustfully or treats him / her \" only as an object to satisfy instinct \" ."
        },
        {
            ## should we try to solve "if only"?
            "marker": "if",
            "output": None,
            "previous_sentence": "But the young ones , sir — I 'm sure they 'll understand when I explain to them why you 're so upset .",
            "sentence": "We 'll do everything we can to protect you from now on , if only you 'll forgive us , and we 'll be sure to let you know when anything good is going to happen ! \""
        },
    ]
        
    print("{} cases are weird and I can't figure out how to handle them. :(".format(len(curious_cases)))
    print("{} of those incorrectly return None".format(len([c for c in curious_cases if depparse_ssplit(c["sentence"], c["previous_sentence"], c["marker"])==None])))
    print("{} parsable cases are being tested".format(len(test_items)))
    curious=False
    if curious:
        print("running curious cases...")
        for item in curious_cases:
            print("====================")
            print(item["sentence"])
            output = depparse_ssplit(item["sentence"], item["previous_sentence"], item["marker"])
            print(output)
        print("====================")
        print("====================")
        print("====================")
    marker_accuracy=False
    if marker_accuracy:
        markers = set([c["marker"] for c in test_items + curious_cases])
        for marker in markers:
            correct = len([c for c in test_items if c["marker"]==marker])
            incorrect = len([c for c in curious_cases if c["marker"]==marker])
            accuracy = float(correct) / (correct+incorrect)
            print("{} ~ {}".format(marker, accuracy))


    # n_tests = 79
    # i = 0
    failures = 0
    print("running tests...")

    for item in test_items:
        # if i < n_tests:
            output = depparse_ssplit(item["sentence"], item["previous_sentence"], item["marker"])
            try:
                assert(output == item["output"])
            except AssertionError:
                print("====== TEST FAILED ======" + "\nsentence: " + item["sentence"] + "\nmarker: " + item["marker"] + "\nactual output: " + str(output) + "\ndesired output: " + str(item["output"]))
                failures += 1
        # else:
        #     print("====================")
        #     print(item["sentence"])
        #     output = depparse_ssplit(item["sentence"], item["previous_sentence"], item["marker"])
        #     print(output)
        # i += 1

    if failures==0:
        print("All tests passed.")

if __name__ == '__main__':
    args = setup_args()
    test()

